import os, re, httpx
from fastapi import FastAPI, Request, Response

app = FastAPI(title="RouteLLM Proxy", version="1.0.0")

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://nexus.assistant.lamanify.com/v1").rstrip("/")
NEXUS_KEY = os.environ.get("NEXUS_API_KEY", "sk-726c286c1beba453db47eb465f80b439")

ROUTINE_KEYWORDS = [
    r"\bping\b", r"\bstatus\b", r"\bhello\b", r"\bhi\b", r"\bthanks\b",
    r"\bdate\b", r"\btime\b", r"\blist\b", r"\bcheck\b", r"\bheartbeat\b"
]
THINK_KEYWORDS = [
    r"\baudit\b", r"\breason\b", r"\barchitecture\b", r"\bplan\b",
    r"\bdeep research\b", r"\broot cause\b", r"\brefactor\b", r"\banalyze\b",
    r"\bbenchmark\b", r"\bstrategy\b", r"\bsecurity\b"
]

def classify_prompt(text: str) -> str:
    text_lower = text.lower()
    for kw in THINK_KEYWORDS:
        if re.search(kw, text_lower):
            return "think"
    if len(text.split()) < 8:
        for kw in ROUTINE_KEYWORDS:
            if re.search(kw, text_lower):
                return "routine"
    return "chat"

@app.get("/health")
def health():
    return {"status": "ok", "service": "routellm-proxy"}

@app.get("/v1/models")
async def list_models():
    # Mirror models from upstream
    headers = {"Authorization": f"Bearer {NEXUS_KEY}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{UPSTREAM}/models", headers=headers)
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_req = body.get("model", "auto")
    
    # Auto-classify if model is auto, default, chat, or routellm
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
        
        chosen = classify_prompt(user_text) if user_text else "chat"
        print(f"[RouteLLM] User prompt: {user_text[:50]!r} -> Routed to: {chosen}")
        body["model"] = chosen

    headers = {
        "Authorization": f"Bearer {NEXUS_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{UPSTREAM}/chat/completions",
            json=body,
            headers=headers
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )
