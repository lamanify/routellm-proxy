import os, json, httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response

app = FastAPI(title="RouteLLM Proxy", version="1.2.0")

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://nexus.assistant.lamanify.com/v1").rstrip("/")
NEXUS_KEY = os.environ.get("NEXUS_API_KEY", "")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "CLD/cf/@cf/meta/llama-3.1-8b-instruct-fp8-fast")

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

async def classify_prompt_llm(client: httpx.AsyncClient, text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "chat"
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
                "temperature": 0.0
            },
            timeout=3.0
        )
        if resp.status_code == 200:
            raw = resp.text.replace("data: [DONE]", "").strip()
            data = json.loads(raw)
            msg = data["choices"][0]["message"]["content"]
            route = json.loads(msg).get("route", "").lower().strip()
            if route in ["routine", "chat", "think"]:
                return route
    except Exception as e:
        print(f"[RouteLLM] Classifier fallback on error: {e}")
    return "chat"

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
            chosen = await classify_prompt_llm(client, user_text) if user_text else "chat"
            print(f"[RouteLLM] User prompt: {user_text[:60]!r} -> Routed to: {chosen}")
            body["model"] = chosen

    is_stream = bool(body.get("stream", False))
    headers = {
        "Authorization": f"Bearer {NEXUS_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity"
    }
    
    if is_stream:
        client = httpx.AsyncClient(timeout=180.0)
        req = client.build_request("POST", f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        resp = await client.send(req, stream=True)
        
        async def stream_generator():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_generator(),
            status_code=resp.status_code,
            headers=clean_headers(resp.headers),
            media_type=resp.headers.get("content-type", "text/event-stream")
        )
    else:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{UPSTREAM}/chat/completions", json=body, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=clean_headers(resp.headers),
                media_type=resp.headers.get("content-type", "application/json")
            )
