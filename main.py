import os, re, json, time, asyncio, hashlib, httpx
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://nexus.assistant.lamanify.com/v1").rstrip("/")
NEXUS_KEY = os.environ.get("NEXUS_API_KEY", "")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "CLD/cf/@cf/meta/llama-3.1-8b-instruct-fp8-fast")
# Primary classifier: Jev (TypeSafe decisions model) via OpenRouter's decisions endpoint. It returns one
# typed label with a confidence; errors and low-confidence answers fall back to the ROUTER_MODEL LLM.
# Without OPENROUTER_API_KEY the LLM classifier is used alone.
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.14")
JEV_URL = os.environ.get("JEV_URL", "https://openrouter.ai/api/alpha/decisions")
JEV_MIN_CONFIDENCE = float(os.environ.get("JEV_MIN_CONFIDENCE", "0.35"))
USE_JEV = bool(OPENROUTER_KEY) and os.environ.get("CLASSIFIER_BACKEND", "jev") == "jev"

# Upstream sometimes accepts a streaming request and then sends nothing for minutes.
# Give each attempt FIRST_TOKEN_TIMEOUT seconds to produce its first chunk, then retry.
FIRST_TOKEN_TIMEOUT = float(os.environ.get("FIRST_TOKEN_TIMEOUT", "60"))
STALL_RETRIES = int(os.environ.get("STALL_RETRIES", "2"))
STREAM_READ_TIMEOUT = float(os.environ.get("STREAM_READ_TIMEOUT", "180"))
# If an attempt has no first chunk after HEDGE_AFTER seconds, start the next one alongside it
# and keep whichever answers first (0 disables hedging).
HEDGE_AFTER = float(os.environ.get("HEDGE_AFTER", "15"))
# Very large requests (long agent sessions) can legitimately think for over a minute before the
# first chunk; give them FIRST_TOKEN_TIMEOUT_LARGE instead of killing healthy-but-slow attempts.
FIRST_TOKEN_TIMEOUT_LARGE = float(os.environ.get("FIRST_TOKEN_TIMEOUT_LARGE", "120"))
LARGE_REQUEST_CHARS = int(os.environ.get("LARGE_REQUEST_CHARS", "150000"))
# Hermes's continuation nudge after an output-cap truncation. Route it to the tier with the
# largest output budget: inheriting a low-thinking tier (8k cap) cannot finish a long answer.
TRUNCATION_MARKER = "[System: Your previous response was truncated"
TRUNCATION_TIER = os.environ.get("TRUNCATION_TIER", "think")
# Hermes compaction handoff markers (agent/context_compressor.py) that start a compacted session.
COMPACTION_PREFIXES = ("[CONTEXT COMPACTION", "[PRIOR CONTEXT")


def request_chars(body: dict) -> int:
    """Approximate request size: total characters of message content (text parts only)."""
    total = 0
    for m in body.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(p.get("text") or "") for p in c if isinstance(p, dict))
    return total


def first_token_timeout(body: dict) -> float:
    if request_chars(body) >= LARGE_REQUEST_CHARS:
        return max(FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_TIMEOUT_LARGE)
    return FIRST_TOKEN_TIMEOUT

# Classifier: a request waits at most CLASSIFIER_BUDGET seconds for the route, then uses the
# conversation's last tier (DEFAULT_TIER if none) while the classification finishes in the
# background. One classification gets CLASSIFIER_TIMEOUT seconds in total, with a second
# attempt started alongside the first after CLASSIFIER_HEDGE seconds (0 disables the hedge).
CLASSIFIER_BUDGET = float(os.environ.get("CLASSIFIER_BUDGET", "1.5"))
CLASSIFIER_TIMEOUT = float(os.environ.get("CLASSIFIER_TIMEOUT", "2.5"))
CLASSIFIER_HEDGE = float(os.environ.get("CLASSIFIER_HEDGE", "1.0"))
# Classifier latency grows with input size (measured: ~+1.3s at 120k chars, +3-5s at 250k), so it only
# sees the start and end of a long instruction and the first CONTEXT_CHARS of the quoted parent.
INSTRUCTION_CHARS = int(os.environ.get("CLASSIFIER_INSTRUCTION_CHARS", "1500"))
CONTEXT_CHARS = int(os.environ.get("CLASSIFIER_CONTEXT_CHARS", "500"))
DEFAULT_TIER = os.environ.get("DEFAULT_TIER", "chat")
# Rule-based shortcuts (greetings, continuations, stack traces) that skip the LLM classifier.
PREFILTER = os.environ.get("PREFILTER", "1") != "0"
# Classify the Slack instruction instead of the whole wrapped message, and inherit the
# conversation's last tier for continuations and system notices.
CONTEXT_ROUTING = os.environ.get("CONTEXT_ROUTING", "1") != "0"
# Warn when the model that actually answered does not contain this text ("" disables).
EXPECTED_MODEL = os.environ.get("EXPECTED_MODEL_SUBSTR", "gemini").lower()
MODELS_CACHE_TTL = float(os.environ.get("MODELS_CACHE_TTL", "60"))

TIERS = ("routine", "chat", "think")
if DEFAULT_TIER not in TIERS:
    DEFAULT_TIER = "chat"

# An agent turn makes many calls that all end with the same user message; classify it once.
ROUTE_CACHE_TTL = float(os.environ.get("ROUTE_CACHE_TTL", "1800"))
ROUTE_CACHE_MAX = 5000
route_cache: dict[str, tuple[str, float]] = {}
CONV_TTL = float(os.environ.get("CONV_TTL", "21600"))
CONV_MAX = 5000
conv_state: "OrderedDict[str, tuple[str, float]]" = OrderedDict()  # conversation key -> (last tier, time)
inflight: dict[str, asyncio.Task] = {}
background: set[asyncio.Task] = set()
stats: Counter = Counter()
# Hermes background prompts (skill review, compression) go to "chat" without classifying.
UNCLASSIFIED_CHAT_MARKERS = ("Review the conversation above", "summarization agent")

TURSO_URL = os.environ.get("TURSO_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

CLASSIFIER_PROMPT = """Classify the user's request into one of: routine, chat, think.
routine: greetings, thanks, simple acknowledgments, pings, simple lookups of a link, status or fact
chat: standard chat, explanations, copywriting, typical tasks
think: audits, architecture, code debugging, disaster recovery plans, troubleshooting complex errors, and requests to check, crawl, test or review many pages or systems or to carry out multi-step work
Judge by how much work the request implies, not by how short it is. A context line, if present, is only the message being replied to: use it to understand what the request refers to.

Return JSON: {"route": "routine"|"chat"|"think"}"""

# The same tiers as a Jev Choice question (option name -> description).
JEV_INSTRUCTIONS = ("How much work does the user's request imply? Judge by how much work it implies, not by how "
                    "short it is. A context line, if present, is only the message being replied to.")
JEV_CRITERIA = {
    "routine": "greetings, thanks, simple acknowledgments, pings, simple lookups of a link, status or fact",
    "chat": "standard chat, explanations, copywriting, typical tasks",
    "think": "audits, architecture, code debugging, disaster recovery plans, troubleshooting complex errors, and "
             "requests to check, crawl, test or review many pages or systems or to carry out multi-step work",
}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding", "content-length"
}

http: httpx.AsyncClient  # shared pool (set in lifespan) so Nexus connections stay warm

@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    http = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        transport=httpx.AsyncHTTPTransport(retries=1),  # retries connect errors only
    )
    try:
        yield
    finally:
        for task in list(background):
            task.cancel()
        await http.aclose()

app = FastAPI(title="RouteLLM Proxy", version="1.6.0", lifespan=lifespan)

def clean_headers(headers: httpx.Headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}

def spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    background.add(task)
    task.add_done_callback(background.discard)
    return task

def log_event(**fields):
    """One JSON line per request: route, fallback reason, timings, answering model."""
    print(json.dumps({"evt": "req", "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", **fields},
                     separators=(",", ":"), default=str), flush=True)

def ms_since(t0: float) -> float:
    return round((time.time() - t0) * 1000, 1)

async def log_to_turso(prompt_preview: str, route: str, latency_ms: float, client_ip: str = "",
                       status: str = "success", extra: dict | None = None):
    if not TURSO_URL or not TURSO_TOKEN:
        return
    payload = {
        "prompt": prompt_preview,
        "route": route,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "model": ROUTER_MODEL,
        **(extra or {}),
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
            {"type": "text", "value": status},
            {"type": "text", "value": json.dumps(payload)},
            {"type": "text", "value": str(int(datetime.now(timezone.utc).timestamp()))}
        ]
    }
    try:
        await http.post(
            TURSO_URL,
            headers={"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"},
            json={"requests": [{"type": "execute", "stmt": stmt}]},
            timeout=3.0,
        )
    except Exception as e:
        print(f"[RouteLLM] Turso telemetry background error: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Reading the conversation: what is the user actually asking?
# ---------------------------------------------------------------------------

def message_text(m: dict) -> str:
    content = m.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return ""

# Hermes-injected blocks that are not the user's words.
MEMORY_RE = re.compile(r"<memory-context>.*?(?:</memory-context>|\Z)", re.S)
SYSTEM_NOTE_RE = re.compile(r"\[System note.*?(?:\n\s*\n|\Z)", re.S | re.I)
# Messages that carry no new request: system notes, delegation/iteration notices, compaction handoffs.
NOTICE_RE = re.compile(
    r"^\s*\[(?:System\b|ASYNC DELEGATION|CONTEXT COMPACTION|STILL IN PROGRESS|Context from the interrupted"
    r"|[^\]\n]{0,60}\b(?:tool[- ]iteration|iteration (?:limit|budget)|max(?:imum)? iterations))", re.I)
REPLY_RE = re.compile(r'^\s*\[Replying to:\s*"(?P<ctx>.*?)"\]\s*(?P<rest>.*)$', re.S)
REPLY_WITH_SENDER_RE = re.compile(r'^\s*\[Replying to:\s*"(?P<ctx>.*?)"\]\s*(?P<rest>\[[^\]\n]{1,80}\].*)$', re.S)
SENDER_RE = re.compile(r"^\s*\[[^\]\n|]{1,60}(?:\s*\|\s*Slack user\s*<@\w+>)?\]\s*")
OOB_RE = re.compile(r"^\s*\[OUT-OF-BAND USER MESSAGE")
THREAD_RE = re.compile(r"^\s*\[Thread context\b[^\]\n]*\]\s*", re.I)

def split_message(text: str) -> tuple[str, str]:
    """Return (instruction, context). instruction == "" means the message carries no new request."""
    text = MEMORY_RE.sub("", text)
    if NOTICE_RE.match(text):
        return "", ""
    text = SYSTEM_NOTE_RE.sub("", text).strip()
    context = ""
    if OOB_RE.match(text):
        lines = text.split("\n")[1:]
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith(("Gateway message origin", "{", "["))):
            lines.pop(0)
        text = "\n".join(lines).strip()
    elif THREAD_RE.match(text):
        parts = THREAD_RE.sub("", text, count=1).split("\n\n")
        text, context = (parts[-1], "\n\n".join(parts[:-1])) if len(parts) > 1 else ("", "")
    m = REPLY_WITH_SENDER_RE.match(text) or REPLY_RE.match(text)
    if m:
        context, text = m.group("ctx"), m.group("rest")
    return SENDER_RE.sub("", text, count=1).strip(), context.strip()

# --- rule-based prefilter (obvious cases, no LLM call) ---
SOCIAL_RE = re.compile(
    r"^(?:hi+|hello|hey+|yo|hiya|good (?:morning|afternoon|evening)|ping|pong|test|testing"
    r"|thanks?(?: you)?(?: (?:a lot|so much|very much|again|received|noted|man|mate|bro))?|thank you(?: so much| very much)?|thx|ty|cheers|many thanks"
    r"|got it|noted|received|understood|cool|nice|great|awesome|perfect|will do|(?:ok|okay),? thanks?)"
    r"(?:\s+(?:there|all|team|everyone|guys))?(?:\s+[a-z]{2,12})?$")
CONTINUE_WORDS = frozenset("""continue go on ahead proceed carry keep going yes yep yeah yup ok okay sure approved approve confirmed confirm
    do it that this ship deploy retry try again fix apply run please pls now then and the plan lgtm looks good sounds to me""".split())
CONTINUE_SIGNALS = frozenset("""continue go proceed carry keep yes yep yeah yup ok okay sure approved approve confirmed confirm
    do ship deploy retry try fix apply run lgtm looks sounds""".split())
STACK_RE = re.compile(
    r"Traceback \(most recent call last\)|\bStack trace:|PHP (?:Fatal|Parse) error|Fatal error: Uncaught|Exception in thread"
    r"|\bpanic: |Segmentation fault|OOMKilled|File \"[^\"\n]+\", line \d+|(?:^|\n)\s*at [\w.$<>]+ \([^)\n]+:\d+:\d+\)"
    r"|(?:^|\n)#\d+ [^\n]*\.php")

def prefilter(instruction: str, context: str) -> tuple[str, str] | None:
    """Cheap rules for obvious cases. Returns ("tier", <tier>) or ("inherit", <tier if no history>)."""
    flat = re.sub(r"[^\w\s]|_", " ", instruction.lower())
    flat = re.sub(r"\s+", " ", flat).strip()
    words = flat.split()
    if not words:
        return None
    if len(words) <= 6 and STACK_RE.search(context) and len(instruction) <= 60 and not SOCIAL_RE.match(flat):
        return "tier", "think"
    if STACK_RE.search(instruction):
        return "tier", "think"
    if len(words) <= 5 and SOCIAL_RE.match(flat):
        return "tier", "routine"
    if len(words) <= 6 and all(w in CONTINUE_WORDS for w in words) and any(w in CONTINUE_SIGNALS for w in words):
        return "inherit", "routine" if words[0] in ("lgtm", "looks", "sounds") else DEFAULT_TIER
    return None

# ---------------------------------------------------------------------------
# Route state: per-conversation last tier, per-conversation-and-message cache
# ---------------------------------------------------------------------------

def conversation_key(messages: list) -> str:
    """All requests arrive from one proxy IP, so identify a conversation from its body.

    Only the start of the system prompt and of the first user message is hashed, so a timestamp or other
    per-call text further down does not make every call look like a new conversation."""
    system = next((message_text(m) for m in messages if m.get("role") in ("system", "developer")), "")
    first_user = next((message_text(m) for m in messages if m.get("role") == "user"), "")
    # After a Hermes compaction the first user message is a fixed preamble (>1000 chars) plus the
    # session's own summary, so its first 1000 chars are identical for every compacted session of a
    # bot. Hash the whole message there: the summary is unique per session and stable until the
    # next compaction.
    head = first_user if first_user.startswith(COMPACTION_PREFIXES) else first_user[:1000]
    return hashlib.sha1(f"{system[:1000]}\x00{head}".encode()).hexdigest()[:12]

def conv_last_tier(conv: str) -> str | None:
    entry = conv_state.get(conv)
    if entry and time.time() - entry[1] < CONV_TTL:
        return entry[0]
    return None

def conv_set(conv: str, tier: str):
    conv_state[conv] = (tier, time.time())
    conv_state.move_to_end(conv)
    while len(conv_state) > CONV_MAX:
        conv_state.popitem(last=False)

def cache_put(key: str, tier: str):
    if len(route_cache) >= ROUTE_CACHE_MAX:
        cutoff = time.time() - ROUTE_CACHE_TTL
        for k in [k for k, (_, t) in route_cache.items() if t < cutoff] or list(route_cache)[:ROUTE_CACHE_MAX // 2]:
            del route_cache[k]
    route_cache[key] = (tier, time.time())

# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

def classifier_input(instruction: str, context: str) -> str:
    if len(instruction) > INSTRUCTION_CHARS:
        head = INSTRUCTION_CHARS * 2 // 3
        instruction = instruction[:head] + " ... " + instruction[-(INSTRUCTION_CHARS - head):]
    text = f"Query: {instruction}"
    if context:
        text += f"\nContext (the message being replied to): {context[:CONTEXT_CHARS]}"
    return text

async def jev_call(instruction: str, context: str) -> tuple[str | None, float, str]:
    """One Jev decision. Returns (route, confidence, "ok") or (None, 0.0, reason)."""
    try:
        resp = await http.post(
            JEV_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": JEV_MODEL, "state": classifier_input(instruction, context),
                  "questions": {"route": {"type": "choice", "instructions": JEV_INSTRUCTIONS, "criteria": JEV_CRITERIA}}},
            timeout=CLASSIFIER_TIMEOUT,
        )
    except httpx.TimeoutException:
        return None, 0.0, "jev_timeout"
    except httpx.HTTPError as e:
        return None, 0.0, f"jev_transport_{type(e).__name__}"
    if resp.status_code != 200:
        return None, 0.0, f"jev_http_{resp.status_code}"
    try:
        answer = resp.json()["answers"]["route"]
        route, confidence = str(answer["choice"]).lower().strip(), float(answer.get("confidence") or 0.0)
    except (ValueError, KeyError, TypeError):
        return None, 0.0, "jev_invalid"
    return (route, confidence, "ok") if route in TIERS else (None, 0.0, "jev_invalid_route")

async def classifier_call(instruction: str, context: str) -> tuple[str | None, str]:
    """One classification: Jev first when configured; the LLM classifier answers when Jev fails or is unsure
    (confidence below JEV_MIN_CONFIDENCE). An unsure Jev answer still beats no answer."""
    if not USE_JEV:
        return await llm_call(instruction, context)
    route, confidence, reason = await jev_call(instruction, context)
    if route and confidence >= JEV_MIN_CONFIDENCE:
        stats["classifier_jev"] += 1
        return route, "ok"
    llm_route, llm_reason = await llm_call(instruction, context)
    if llm_route:
        stats["classifier_llm_after_" + ("jev_unsure" if route else reason)] += 1
        return llm_route, "ok"
    if route:
        stats["classifier_jev_unsure_kept"] += 1
        return route, "ok"
    return None, f"{reason}+{llm_reason}"

async def llm_call(instruction: str, context: str) -> tuple[str | None, str]:
    """One LLM classifier request. Returns (route, "ok") or (None, reason)."""
    try:
        resp = await http.post(
            f"{UPSTREAM}/chat/completions",
            headers={"Authorization": f"Bearer {NEXUS_KEY}", "Content-Type": "application/json"},
            json={
                "model": ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": classifier_input(instruction, context)}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 15,
                "temperature": 0.0,
                "stream": False
            },
            timeout=CLASSIFIER_TIMEOUT,
        )
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as e:
        return None, f"transport_{type(e).__name__}"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"
    try:
        raw = resp.text.replace("data: [DONE]", "").strip()
        route = json.loads(json.loads(raw)["choices"][0]["message"]["content"]).get("route", "")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None, "invalid_json"
    route = str(route).lower().strip()
    return (route, "ok") if route in TIERS else (None, "invalid_route")

async def classify_hedged(instruction: str, context: str) -> tuple[str | None, str, float, int]:
    """Classify within CLASSIFIER_TIMEOUT, hedging after CLASSIFIER_HEDGE. Returns (route, reason, ms, attempts)."""
    t0 = time.time()
    deadline = t0 + CLASSIFIER_TIMEOUT
    tasks = {asyncio.create_task(classifier_call(instruction, context))}
    attempts, last_reason = 1, "timeout"
    can_hedge = CLASSIFIER_HEDGE > 0
    try:
        while tasks:
            now = time.time()
            if now >= deadline:
                break
            wait = deadline - now
            if can_hedge:
                wait = min(wait, max(0.0, t0 + CLASSIFIER_HEDGE - now))
            done, _ = await asyncio.wait(tasks, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tasks.discard(task)
                route, reason = task.result()
                if route:
                    return route, "ok", ms_since(t0), attempts
                last_reason = reason
            if can_hedge and (not done and time.time() >= t0 + CLASSIFIER_HEDGE - 0.01 or not tasks):
                can_hedge = False  # one extra attempt: on the hedge timer, or at once if the first one failed
                attempts += 1
                tasks.add(asyncio.create_task(classifier_call(instruction, context)))
    finally:
        for task in tasks:
            task.cancel()
    return None, last_reason, ms_since(t0), attempts

async def run_classification(key: str, conv: str, instruction: str, context: str, preview: str, client_ip: str):
    """Classify and record the outcome. Runs as a task so it finishes even if the request stopped waiting."""
    try:
        route, reason, ms, attempts = await classify_hedged(instruction, context)
        if route:
            stats["classifier_ok"] += 1
            cache_put(key, route)  # only real answers are cached, never fallbacks
            conv_set(conv, route)
        else:
            stats[f"classifier_fail_{reason}"] += 1
            print(f"[RouteLLM] Classifier failed ({reason}) after {ms}ms, {attempts} attempt(s)")
        spawn(log_to_turso(preview, route or conv_last_tier(conv) or DEFAULT_TIER, ms, client_ip,
                           status="success" if route else "fallback",
                           extra={"src": "llm" if route else "fallback", "reason": reason, "attempts": attempts,
                                  "input_chars": len(instruction) + len(context)}))
        return route, reason, ms, attempts
    finally:
        inflight.pop(key, None)

async def route_request(messages: list, client_ip: str = "") -> dict:
    """Pick a tier for a chat request. Never waits more than CLASSIFIER_BUDGET for the LLM."""
    conv = conversation_key(messages)
    last_user = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
    d = {"tier": DEFAULT_TIER, "src": "default", "reason": None, "classify_ms": 0.0, "conv": conv, "prompt": last_user[:60],
         "input_chars": len(last_user)}
    if not last_user.strip():
        return d
    if any(marker in last_user for marker in UNCLASSIFIED_CHAT_MARKERS):
        return {**d, "tier": "chat", "src": "background"}
    if last_user.lstrip().startswith(TRUNCATION_MARKER) and TRUNCATION_TIER in TIERS:
        return {**d, "tier": TRUNCATION_TIER, "src": "rule", "reason": "truncation_continuation"}

    previous = conv_last_tier(conv)
    if CONTEXT_ROUTING:
        instruction, context = split_message(last_user)
        d["prompt"] = (instruction or last_user)[:60]
        if not instruction:
            return {**d, "tier": previous or DEFAULT_TIER, "src": "inherit", "reason": "notice"}
    else:
        instruction, context = last_user.strip(), ""
    if PREFILTER:
        hit = prefilter(instruction, context)
        if hit and hit[0] == "tier":
            if hit[1] != "routine":
                conv_set(conv, hit[1])
            return {**d, "tier": hit[1], "src": "rule"}
        if hit and hit[0] == "inherit" and (previous or not context):
            return {**d, "tier": previous or hit[1], "src": "inherit", "reason": "continuation"}

    key = hashlib.sha256(f"{conv}\x00{instruction}\x00{context[:300]}".encode()).hexdigest()
    cached = route_cache.get(key)
    if cached and time.time() - cached[1] < ROUTE_CACHE_TTL:
        conv_set(conv, cached[0])
        return {**d, "tier": cached[0], "src": "cache"}

    task = inflight.get(key)
    if task is None:
        task = inflight[key] = spawn(run_classification(key, conv, instruction, context, instruction[:200], client_ip))
    done, _ = await asyncio.wait({task}, timeout=CLASSIFIER_BUDGET)
    fallback = previous or DEFAULT_TIER
    if not done:
        stats["fallback_budget"] += 1
        return {**d, "tier": fallback, "src": "fallback", "reason": "budget", "classify_ms": round(CLASSIFIER_BUDGET * 1000, 1)}
    try:
        route, reason, ms, _ = task.result()
    except Exception as e:  # classification task crashed: treat as a failed classification
        route, reason, ms = None, f"error_{type(e).__name__}", 0.0
    if route:
        return {**d, "tier": route, "src": "llm", "classify_ms": ms}
    stats[f"fallback_{reason}"] += 1
    return {**d, "tier": fallback, "src": "fallback", "reason": reason, "classify_ms": ms}

def start_attempt(body: dict, headers: dict, attempt: int) -> dict:
    """Start one upstream streaming attempt as a task that resolves to (chunks, first_chunk)."""
    state = {"attempt": attempt, "client": httpx.AsyncClient(timeout=STREAM_READ_TIMEOUT), "resp": None, "t0": time.time()}

    async def first_chunk():
        req = state["client"].build_request("POST", f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        state["resp"] = await state["client"].send(req, stream=True)
        chunks = state["resp"].aiter_raw()
        return chunks, await anext(chunks, b"")

    state["task"] = asyncio.create_task(asyncio.wait_for(first_chunk(), first_token_timeout(body)))
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
            "message": f"Upstream gave no first chunk within the first-token timeout "
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


MODEL_FIELD_RE = re.compile(rb'"model"\s*:\s*"([^"]+)"')

def model_from_bytes(raw: bytes) -> str | None:
    m = MODEL_FIELD_RE.search(raw)
    return m.group(1).decode("utf-8", "replace") if m else None

def check_model(rec: dict, model: str | None):
    """Record which model answered and flag answers from outside the expected family (e.g. fallbacks)."""
    rec["resp_model"] = model
    if EXPECTED_MODEL and model and EXPECTED_MODEL not in model.lower():
        rec["model_warn"] = True
        stats["unexpected_model"] += 1
        print(f"[RouteLLM] WARN answered by {model!r}, not a {EXPECTED_MODEL} model (tier {rec.get('tier')}): likely an upstream fallback")

models_cache: dict = {"entry": None, "at": 0.0}
models_lock = asyncio.Lock()

@app.get("/health")
def health():
    return {"status": "ok", "service": "routellm-proxy"}

@app.get("/stats")
def get_stats():
    return {"counters": dict(stats), "route_cache": len(route_cache), "conversations": len(conv_state), "inflight": len(inflight)}

@app.get("/v1/models")
async def list_models():
    async with models_lock:  # one upstream fetch at a time; the rest read the cache
        entry = models_cache["entry"]
        if entry is None or time.time() - models_cache["at"] >= MODELS_CACHE_TTL:
            try:
                resp = await http.get(f"{UPSTREAM}/models", headers={"Authorization": f"Bearer {NEXUS_KEY}", "Accept-Encoding": "identity"})
            except httpx.HTTPError:
                if entry is None:
                    raise
                stats["models_stale"] += 1  # upstream down: serve the last good list
            else:
                entry = (resp.content, resp.status_code, clean_headers(resp.headers), resp.headers.get("content-type", "application/json"))
                if resp.status_code == 200:
                    models_cache["entry"], models_cache["at"] = entry, time.time()
    content, status, headers, media_type = entry
    return Response(content=content, status_code=status, headers=headers, media_type=media_type)

@app.post("/v1/route")
async def route_only(request: Request):
    """Routing decision for a chat request body, without calling the model (used by run_evals.py)."""
    body = await request.json()
    d = await route_request(body.get("messages") or [], request.client.host if request.client else "")
    return {"route": d["tier"], "src": d["src"], "reason": d["reason"], "classify_ms": d["classify_ms"], "conversation": d["conv"]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t_start = time.time()
    body = await request.json()
    model_req = body.get("model", "auto")
    chosen = "chat"
    rec = {"stream": bool(body.get("stream", False)), "model_req": model_req}

    if model_req in ["auto", "default", "chat", "routellm"]:
        d = await route_request(body.get("messages") or [], request.client.host if request.client else "")
        chosen = d["tier"]
        rec.update(conv=d["conv"], tier=chosen, route_src=d["src"], fallback_reason=d["reason"] if d["src"] == "fallback" else None,
                   classify_ms=d["classify_ms"], input_chars=d["input_chars"], prompt=d["prompt"])
        body["model"] = chosen
    else:
        rec.update(tier=model_req, route_src="explicit")
    latency_ms = rec.get("classify_ms", 0.0)

    is_stream = rec["stream"]
    headers = {
        "Authorization": f"Bearer {NEXUS_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity"
    }

    def finish(status: int, **extra):
        log_event(**rec, status=status, total_ms=ms_since(t_start), **extra)

    t_upstream = time.time()
    if is_stream:
        try:
            client, resp, chunks, first, attempt = await open_upstream_stream(body, headers)
        except (asyncio.TimeoutError, httpx.TransportError) as e:
            finish(504 if isinstance(e, asyncio.TimeoutError) else 502, upstream_error=type(e).__name__)
            return stall_error(e, chosen)
        rec.update(attempts=attempt, ttft_ms=ms_since(t_start), upstream_ttft_ms=ms_since(t_upstream))
        check_model(rec, model_from_bytes(first))

        async def stream_generator():
            seen = first
            completed = False
            try:
                if first:
                    yield first
                async for chunk in chunks:
                    if rec["resp_model"] is None and len(seen) < 8192:
                        seen += chunk
                        check_model(rec, model_from_bytes(seen))
                    yield chunk
                completed = True
            finally:
                await resp.aclose()
                await client.aclose()
                finish(resp.status_code, completed=completed)

        resp_headers = clean_headers(resp.headers)
        resp_headers["X-RouteLLM-Choice"] = chosen
        resp_headers["X-RouteLLM-Latency"] = f"{latency_ms}ms"
        resp_headers["X-RouteLLM-Attempts"] = str(attempt)
        resp_headers["X-RouteLLM-Source"] = rec.get("route_src", "")
        if rec.get("resp_model"):
            resp_headers["X-RouteLLM-Model"] = rec["resp_model"]

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
            finish(504 if isinstance(e, asyncio.TimeoutError) else 502, upstream_error=type(e).__name__)
            return stall_error(e, chosen)
        rec.update(attempts=attempt, ttft_ms=ms_since(t_start), upstream_ttft_ms=ms_since(t_upstream))
        try:
            raw = first + b"".join([chunk async for chunk in chunks])
        except httpx.TransportError as e:
            finish(502, upstream_error=f"mid_stream_{type(e).__name__}")
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Upstream stream broke mid-response ({type(e).__name__})",
                                   "type": "upstream_error"}},
                headers={"X-RouteLLM-Choice": chosen},
            )
        finally:
            await resp.aclose()
            await client.aclose()

        is_sse = "text/event-stream" in resp.headers.get("content-type", "")
        completion = sse_to_completion(raw) if resp.status_code == 200 and is_sse else None
        check_model(rec, (completion or {}).get("model") or model_from_bytes(raw[:8192]))
        resp_headers = {
            "X-RouteLLM-Choice": chosen,
            "X-RouteLLM-Latency": f"{latency_ms}ms",
            "X-RouteLLM-Attempts": str(attempt),
            "X-RouteLLM-Source": rec.get("route_src", ""),
        }
        if rec.get("resp_model"):
            resp_headers["X-RouteLLM-Model"] = rec["resp_model"]
        finish(resp.status_code)
        if completion is not None:
            return JSONResponse(completion, headers=resp_headers)
        return Response(
            content=raw,
            status_code=resp.status_code,
            headers={**clean_headers(resp.headers), **resp_headers},
            media_type=resp.headers.get("content-type", "application/json")
        )
