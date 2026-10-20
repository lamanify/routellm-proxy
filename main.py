import os, json, time, asyncio, hashlib, httpx
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

app = FastAPI(title="RouteLLM Proxy", version="1.4.0")

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://nexus.assistant.lamanify.com/v1").rstrip("/")
NEXUS_KEY = os.environ.get("NEXUS_API_KEY", "")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "CLD/cf/@cf/meta/llama-3.1-8b-instruct-fp8-fast")

# Upstream sometimes accepts a streaming request and then sends nothing for minutes.
# Give each attempt FIRST_TOKEN_TIMEOUT seconds to produce its first chunk, then retry.
FIRST_TOKEN_TIMEOUT = float(os.environ.get("FIRST_TOKEN_TIMEOUT", "60"))
STALL_RETRIES = int(os.environ.get("STALL_RETRIES", "2"))
STREAM_READ_TIMEOUT = float(os.environ.get("STREAM_READ_TIMEOUT", "180"))
# If an attempt has no first chunk after HEDGE_AFTER seconds, start the next one alongside it
# and keep whichever answers first (0 disables hedging).
HEDGE_AFTER = float(os.environ.get("HEDGE_AFTER", "15"))

# An agent turn makes many calls that all end with the same user message; classify it once.
ROUTE_CACHE_TTL = float(os.environ.get("ROUTE_CACHE_TTL", "1800"))
ROUTE_CACHE_MAX = 5000
route_cache: dict[str, tuple[str, float]] = {}
# Hermes background prompts (skill review, compression) go to "chat" without classifying.
UNCLASSIFIED_CHAT_MARKERS = ("Review the conversation above", "summarization agent")

TURSO_URL = os.environ.get("TURSO_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

CLASSIFIER_PROMPT = """Classify query into one of: routine, chat, think.
routine: greetings, link queries, short status checks, pings, simple acknowledgments
chat: standard chat, explanations, copywriting, typical tasks
think: audits, architecture, code debugging, disaster recovery plans, troubleshooting complex errors

Return JSON: {"route": "routine"|"chat"|"think"}"""

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding", "content-length"
}

def clean_headers(headers: httpx.Headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}

async def log_to_turso(prompt_preview: str, route: str, latency_ms: float, client_ip: str = ""):
    if not TURSO_URL or not TURSO_TOKEN:
        print(f"[RouteLLM] Turso telemetry skipped: TURSO_URL={bool(TURSO_URL)}, TURSO_TOKEN={bool(TURSO_TOKEN)}")
        return

    if not TURSO_URL or not TURSO_TOKEN:
        return
    payload = {
        "prompt": prompt_preview,
        "route": route,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "model": ROUTER_MODEL
    }
    stmt = {
        "sql": """
        INSERT INTO universal_agent_logbook 
        (agent_signature, task_signature, execution_status, raw_payload_summary, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        "args": [
            {"type": "text", "value": "routellm-proxy"},
            {"type": "text", "value": f"route:{route}"},
            {"type": "text", "value": "success"},
            {"type": "text", "value": json.dumps(payload)},
            {"type": "text", "value": str(int(datetime.now(timezone.utc).timestamp()))}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                TURSO_URL,
                headers={"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"},
                json={"requests": [{"type": "execute", "stmt": stmt}]}
            )
    except Exception as e:
        print(f"[RouteLLM] Turso telemetry background error: {type(e).__name__}: {e}")

async def classify_prompt_llm(client: httpx.AsyncClient, text: str) -> tuple[str, float]:
    cleaned = text.strip()
    if not cleaned:
        return "chat", 0.0
    t0 = time.time()
    try:
        resp = await client.post(
            f"{UPSTREAM}/chat/completions",
            headers={"Authorization": f"Bearer {NEXUS_KEY}", "Content-Type": "application/json"},
            json={
                "model": ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": f"Query: {cleaned}"}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 15,
                "temperature": 0.0,
                "stream": False
            },
            timeout=10.0
        )
        latency = round((time.time() - t0) * 1000, 1)
        if resp.status_code == 200:
            raw = resp.text.replace("data: [DONE]", "").strip()
            data = json.loads(raw)
            msg = data["choices"][0]["message"]["content"]
            route = json.loads(msg).get("route", "").lower().strip()
            if route in ["routine", "chat", "think"]:
                return route, latency
    except Exception as e:
        print(f"[RouteLLM] Classifier fallback on error: {type(e).__name__}: {e}")
    latency = round((time.time() - t0) * 1000, 1)
    return "chat", latency

def start_attempt(body: dict, headers: dict, attempt: int) -> dict:
    """Start one upstream streaming attempt as a task that resolves to (chunks, first_chunk)."""
    state = {"attempt": attempt, "client": httpx.AsyncClient(timeout=STREAM_READ_TIMEOUT), "resp": None, "t0": time.time()}

    async def first_chunk():
        req = state["client"].build_request("POST", f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        state["resp"] = await state["client"].send(req, stream=True)
        chunks = state["resp"].aiter_raw()
        return chunks, await anext(chunks, b"")

    state["task"] = asyncio.create_task(asyncio.wait_for(first_chunk(), FIRST_TOKEN_TIMEOUT))
    return state

async def close_attempt(state: dict):
    state["task"].cancel()
    try:
        await state["task"]
    except BaseException:
        pass
    if state["resp"] is not None:
        await state["resp"].aclose()
    await state["client"].aclose()

async def open_upstream_stream(body: dict, headers: dict):
    """Send a streaming request, retrying and hedging attempts that produce no first chunk in time.

    Returns (client, resp, chunks, first_chunk, attempt); the caller closes resp and client.
    """
    attempts = 1 + STALL_RETRIES
    hedge_after = HEDGE_AFTER if 0 < HEDGE_AFTER < FIRST_TOKEN_TIMEOUT else None
    running = [start_attempt(body, headers, 1)]
    started = 1
    last_error = None
    try:
        while running:
            done, _ = await asyncio.wait(
                [s["task"] for s in running],
                timeout=hedge_after if started < attempts else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                started += 1
                print(f"[RouteLLM] No first chunk after {hedge_after:.0f}s; hedging with attempt {started}/{attempts}")
                running.append(start_attempt(body, headers, started))
                continue
            for state in [s for s in running if s["task"] in done]:
                running.remove(state)
                try:
                    chunks, first = state["task"].result()
                except (asyncio.TimeoutError, httpx.TransportError) as e:
                    last_error = e
                    print(f"[RouteLLM] No first chunk after {time.time() - state['t0']:.1f}s "
                          f"(attempt {state['attempt']}/{attempts}, {type(e).__name__})")
                    await close_attempt(state)
                    if started < attempts:
                        started += 1
                        running.append(start_attempt(body, headers, started))
                    continue
                if state["attempt"] > 1:
                    print(f"[RouteLLM] First chunk after {time.time() - state['t0']:.1f}s on attempt {state['attempt']}/{attempts}")
                return state["client"], state["resp"], chunks, first, state["attempt"]
        raise last_error
    finally:
        for state in running:  # losers: free their upstream connections
            await close_attempt(state)

def stall_error(e: Exception, chosen: str) -> JSONResponse:
    timed_out = isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException))
    return JSONResponse(
        status_code=504 if timed_out else 502,
        content={"error": {
            "message": f"Upstream gave no first chunk within {FIRST_TOKEN_TIMEOUT:.0f}s "
                       f"on {1 + STALL_RETRIES} attempts ({type(e).__name__})",
            "type": "upstream_timeout" if timed_out else "upstream_error",
        }},
        headers={"X-RouteLLM-Choice": chosen},
    )

def sse_to_completion(raw: bytes) -> dict | None:
    """Merge a stream of chat.completion.chunk events into one chat.completion (None if none parsed)."""
    completion = {"object": "chat.completion"}
    choices = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
            continue
        try:
            chunk = json.loads(line[5:])
        except ValueError:
            continue
        for key in ("id", "created", "model", "system_fingerprint"):
            if chunk.get(key) is not None:
                completion.setdefault(key, chunk[key])
        if chunk.get("usage"):
            completion["usage"] = chunk["usage"]
        for c in chunk.get("choices") or []:
            choice = choices.setdefault(c.get("index", 0), {"message": {"role": "assistant"}, "calls": {}, "finish_reason": None})
            for key, value in (c.get("delta") or {}).items():
                if key == "tool_calls":
                    calls = choice["calls"]
                    for tc in value or []:
                        idx = tc.get("index")
                        if idx is None:  # no index: a new id starts a new call, otherwise continue the last one
                            idx = len(calls) if tc.get("id") or not calls else max(calls)
                        call = calls.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                        call["id"] = tc.get("id") or call["id"]
                        call["type"] = tc.get("type") or call["type"]
                        fn = tc.get("function") or {}
                        call["function"]["name"] = call["function"]["name"] or fn.get("name") or ""
                        call["function"]["arguments"] += fn.get("arguments") or ""
                elif key == "role":
                    choice["message"]["role"] = value or "assistant"
                elif isinstance(value, str):
                    choice["message"][key] = choice["message"].get(key, "") + value
            if c.get("finish_reason"):
                choice["finish_reason"] = c["finish_reason"]
    if not choices:
        return None
    completion["choices"] = []
    for idx, choice in sorted(choices.items()):
        message = choice["message"]
        if choice["calls"]:
            message["tool_calls"] = [call for _, call in sorted(choice["calls"].items())]
            message["content"] = message.get("content") or None
        else:
            message.setdefault("content", "")
        completion["choices"].append({"index": idx, "message": message, "finish_reason": choice["finish_reason"] or "stop"})
    return completion

@app.get("/health")
def health():
    return {"status": "ok", "service": "routellm-proxy"}

@app.get("/v1/models")
async def list_models():
    headers = {
        "Authorization": f"Bearer {NEXUS_KEY}",
        "Accept-Encoding": "identity"
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{UPSTREAM}/models", headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=clean_headers(resp.headers),
            media_type=resp.headers.get("content-type", "application/json")
        )

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_req = body.get("model", "auto")
    chosen = "chat"
    latency_ms = 0.0
    
    if model_req in ["auto", "default", "chat", "routellm"]:
        messages = body.get("messages", [])
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_text += item.get("text", "") + " "
                break
        
        key = hashlib.sha256(user_text.encode()).hexdigest()
        cached = route_cache.get(key)
        if any(marker in user_text for marker in UNCLASSIFIED_CHAT_MARKERS):
            chosen = "chat"
            print(f"[RouteLLM] Prompt: {user_text[:60]!r} -> Routed: chat (background prompt)")
        elif cached and time.time() - cached[1] < ROUTE_CACHE_TTL:
            chosen = cached[0]
            print(f"[RouteLLM] Prompt: {user_text[:60]!r} -> Routed: {chosen} (cached)")
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                chosen, latency_ms = await classify_prompt_llm(client, user_text) if user_text else ("chat", 0.0)
            print(f"[RouteLLM] Prompt: {user_text[:60]!r} -> Routed: {chosen} ({latency_ms}ms)")
            if len(route_cache) >= ROUTE_CACHE_MAX:
                cutoff = time.time() - ROUTE_CACHE_TTL
                for k in [k for k, (_, t) in route_cache.items() if t < cutoff] or list(route_cache)[:ROUTE_CACHE_MAX // 2]:
                    del route_cache[k]
            route_cache[key] = (chosen, time.time())

            # Non-blocking background log to Turso
            asyncio.create_task(
                log_to_turso(
                    prompt_preview=user_text[:200],
                    route=chosen,
                    latency_ms=latency_ms,
                    client_ip=request.client.host if request.client else ""
                )
            )
        body["model"] = chosen

    is_stream = bool(body.get("stream", False))
    headers = {
        "Authorization": f"Bearer {NEXUS_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity"
    }
    
    if is_stream:
        try:
            client, resp, chunks, first, attempt = await open_upstream_stream(body, headers)
        except (asyncio.TimeoutError, httpx.TransportError) as e:
            return stall_error(e, chosen)

        async def stream_generator():
            try:
                if first:
                    yield first
                async for chunk in chunks:
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        resp_headers = clean_headers(resp.headers)
        resp_headers["X-RouteLLM-Choice"] = chosen
        resp_headers["X-RouteLLM-Latency"] = f"{latency_ms}ms"
        resp_headers["X-RouteLLM-Attempts"] = str(attempt)

        return StreamingResponse(
            stream_generator(),
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type", "text/event-stream")
        )
    else:
        # Upstream answers a request without "stream" as SSE, and its non-streaming mode is
        # far slower than streaming. Always stream upstream and merge it into one completion.
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})
        try:
            client, resp, chunks, first, attempt = await open_upstream_stream(body, headers)
        except (asyncio.TimeoutError, httpx.TransportError) as e:
            return stall_error(e, chosen)
        try:
            raw = first + b"".join([chunk async for chunk in chunks])
        except httpx.TransportError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Upstream stream broke mid-response ({type(e).__name__})",
                                   "type": "upstream_error"}},
                headers={"X-RouteLLM-Choice": chosen},
            )
        finally:
            await resp.aclose()
            await client.aclose()

        resp_headers = {
            "X-RouteLLM-Choice": chosen,
            "X-RouteLLM-Latency": f"{latency_ms}ms",
            "X-RouteLLM-Attempts": str(attempt),
        }
        is_sse = "text/event-stream" in resp.headers.get("content-type", "")
        completion = sse_to_completion(raw) if resp.status_code == 200 and is_sse else None
        if completion is not None:
            return JSONResponse(completion, headers=resp_headers)
        return Response(
            content=raw,
            status_code=resp.status_code,
            headers={**clean_headers(resp.headers), **resp_headers},
            media_type=resp.headers.get("content-type", "application/json")
        )
