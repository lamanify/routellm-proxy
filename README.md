# routellm-proxy
Dynamic LLM 3-Tier Classifier Proxy (routine, chat, think) for Lamanify Hermes Fleet

The tiers are three Nexus combos (all Gemini Flash at low / medium / high thinking). A small LLM classifier
(`ROUTER_MODEL`, Cloudflare llama-3.1-8b via Nexus) picks the tier per user message.

## How a request is routed
1. **Background prompts** (skill review, compression) go to `chat` without classification.
2. **Parse the message**: the Slack wrapper (`[Replying to: "<parent>"]`, `[Name | Slack user <@U..>]`, thread context),
   `<memory-context>` blocks and `[System note ...]` blocks are stripped. The instruction is classified; the quoted parent is
   passed to the classifier as context only.
3. **System notices** (`[System note`, `[ASYNC DELEGATION`, `[CONTEXT COMPACTION`, tool-iteration notices, ...) carry no request:
   they inherit the conversation's last tier.
4. **Prefilter** (`PREFILTER=1`): greetings/thanks -> `routine`; stack traces -> `think`; "continue", "yes", "approved, deploy it",
   "fix it" ... inherit the conversation's last tier (with a quoted parent and no history they go to the LLM instead).
5. **Cache** keyed by conversation + instruction + parent. A conversation is identified from the body (hash of the system
   prompt + first user message), because every request arrives from the same proxy IP.
6. **LLM classifier** with a wait budget: the request waits at most `CLASSIFIER_BUDGET`; on a miss it uses the conversation's last
   tier (`DEFAULT_TIER` if none) while the classification finishes in the background and fills the cache. Fallbacks and invalid
   outputs are never cached and are counted in `GET /stats`.

## Endpoints
- `POST /v1/chat/completions`, `GET /v1/models` (cached `MODELS_CACHE_TTL`s), `GET /health`
- `POST /v1/route`: routing decision for a chat body without calling the model (used by `run_evals.py`)
- `GET /stats`: counters (classifier failures by reason, budget fallbacks, unexpected answering models)

Each request prints one JSON line (`"evt":"req"`): route source (`llm|cache|rule|inherit|fallback|background|explicit`),
fallback reason, classify time, TTFT, upstream attempts, the model that actually answered (`resp_model`) and
`model_warn` when it is not a `EXPECTED_MODEL_SUBSTR` (default `gemini`) model, i.e. Nexus used a fallback.
Responses carry `X-RouteLLM-Choice|Source|Latency|Attempts|Model`.

## Environment
| var | default | |
|---|---|---|
| `UPSTREAM_BASE_URL`, `NEXUS_API_KEY`, `ROUTER_MODEL` | | upstream and classifier model |
| `CLASSIFIER_BUDGET` / `CLASSIFIER_TIMEOUT` / `CLASSIFIER_HEDGE` | 1.5 / 2.5 / 1.0 s | wait budget / total timeout / hedge (0 = off) |
| `DEFAULT_TIER` | chat | tier when nothing is known |
| `PREFILTER` / `CONTEXT_ROUTING` | 1 / 1 | `0` turns the rule prefilter / wrapper parsing + inheritance off |
| `ROUTE_CACHE_TTL` / `CONV_TTL` | 1800 / 21600 s | |
| `EXPECTED_MODEL_SUBSTR` | gemini | `""` disables the model warning |
| `MODELS_CACHE_TTL` | 60 s | |
| `FIRST_TOKEN_TIMEOUT`, `STALL_RETRIES`, `HEDGE_AFTER`, `STREAM_READ_TIMEOUT` | 60 / 2 / 15 / 180 | upstream stall handling |
| `TURSO_URL`, `TURSO_AUTH_TOKEN` | | classification telemetry |

## Tests and evals
```
python -m unittest -v test_routing                 # offline: parsing, prefilter, budget, cache, hedge
python run_evals.py --url http://localhost:6060    # classifier-only: eval_suite.json + eval_hermes.json, accuracy and p50/p90
```
`run_evals.py` needs a running proxy that can reach Nexus (`NEXUS_API_KEY`); it never runs a completion.

## Notes on classifier latency (measured 2026-10-21)
- A keep-warm ping was tested and **not** shipped: with synthetic prompts from 70 to 20,000 characters the classifier answered in
  0.4-1.4s after idle gaps of 2, 3, 5, 10 and 15 minutes, the same as a few seconds later, with and without a 30s ping.
- What did correlate with slow classifications in production was input size: ~0.5s at 2k chars, ~1.8s at 120k, 3.6-6s at 250k.
  Hermes' wrapped messages (memory-context, thread context, compaction summaries) are long, so the classifier input is now the
  stripped instruction capped at `CLASSIFIER_INSTRUCTION_CHARS` plus `CLASSIFIER_CONTEXT_CHARS` of the quoted parent.
  Each request logs `input_chars` so this can be confirmed on real traffic.
- The classifier hop occasionally stalls for seconds (all four probes of one burst timed out at 12s): the wait budget, hedge and
  last-tier fallback exist for that.

### If the Nexus -> CLD -> Cloudflare chain stays the weak point
Calling Cloudflare Workers AI directly removes two hops. It needs, from the user: a Cloudflare **account id** and an **API token**
with the Workers AI permission (`CF_ACCOUNT_ID`, `CF_API_TOKEN` env vars). The call would be
`POST https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/ai/run/@cf/meta/llama-3.1-8b-instruct-fp8-fast`
(or the OpenAI-compatible `/ai/v1/chat/completions`) with the same system prompt. Not implemented; the free-tier daily
neuron allowance and per-request rate limits would have to be checked against the classification volume first.
