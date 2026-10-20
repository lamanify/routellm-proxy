import os, json, time, asyncio, httpx
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

async def open_upstream_stream(body: dict, headers: dict):
    """Send a streaming request, retrying attempts that produce no first chunk in time.

    Returns (client, resp, chunks, first_chunk, attempt); the caller closes resp and client.
    """
    attempts = 1 + STALL_RETRIES
    last_error = None
    for attempt in range(1, attempts + 1):
        client = httpx.AsyncClient(timeout=STREAM_READ_TIMEOUT)
        resp = None
        t0 = time.time()

        async def first_chunk():
            nonlocal resp
            req = client.build_request("POST", f"{UPSTREAM}/chat/completions", json=body, headers=headers)
            resp = await client.send(req, stream=True)
            chunks = resp.aiter_raw()
            return chunks, await anext(chunks, b"")

        try:
            chunks, first = await asyncio.wait_for(first_chunk(), FIRST_TOKEN_TIMEOUT)
            if attempt > 1:
                print(f"[RouteLLM] First chunk after {time.time() - t0:.1f}s on attempt {attempt}/{attempts}")
            return client, resp, chunks, first, attempt
        except (asyncio.TimeoutError, httpx.TransportError) as e:
            last_error = e
            retrying = "; retrying" if attempt < attempts else ""
            print(f"[RouteLLM] No first chunk after {time.time() - t0:.1f}s "
                  f"(attempt {attempt}/{attempts}, {type(e).__name__}){retrying}")
            if resp is not None:
                await resp.aclose()
            await client.aclose()
    raise last_error

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
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            chosen, latency_ms = await classify_prompt_llm(client, user_text) if user_text else ("chat", 0.0)
            print(f"[RouteLLM] Prompt: {user_text[:60]!r} -> Routed: {chosen} ({latency_ms}ms)")
            body["model"] = chosen
            
            # Non-blocking background log to Turso
            asyncio.create_task(
                log_to_turso(
                    prompt_preview=user_text[:200],
                    route=chosen,
                    latency_ms=latency_ms,
                    client_ip=request.client.host if request.client else ""
                )
            )

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
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{UPSTREAM}/chat/completions", json=body, headers=headers)
            resp_headers = clean_headers(resp.headers)
            resp_headers["X-RouteLLM-Choice"] = chosen
            resp_headers["X-RouteLLM-Latency"] = f"{latency_ms}ms"
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=resp.headers.get("content-type", "application/json")
            )
