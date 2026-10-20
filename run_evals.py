"""Classifier-only eval: sends each case to POST /v1/route (no model completion is run).

    python run_evals.py [--url http://localhost:6060] [--suite eval_suite.json eval_hermes.json] [--gap SECS]

A case is {"prompt", "expected"} or {"name", "turns": [{"user", "expected"?}, ...]}. A turn's `expected`
is one tier or a list of acceptable tiers. Turns of one case share a conversation (same system prompt and
first message), so continuations can inherit the earlier tier. --gap sleeps between cases to measure the
sparse-traffic latency that production sees.
"""
import argparse, json, os, sys, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("ROUTER_URL", "http://localhost:6060"))
ap.add_argument("--suite", nargs="+", default=["eval_suite.json", "eval_hermes.json"])
ap.add_argument("--gap", type=float, default=0.0)
args = ap.parse_args()
URL = args.url.rstrip("/").removesuffix("/v1")
RUN = uuid.uuid4().hex[:8]

def route(messages):
    req = urllib.request.Request(f"{URL}/v1/route", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": "auto", "messages": messages}).encode())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.load(resp)
    out["wall_ms"] = (time.time() - t0) * 1000
    return out

def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else float("nan")

def run_case(case, system):
    turns = case.get("turns") or [{"user": case["prompt"], "expected": case["expected"]}]
    messages, results = [{"role": "system", "content": system}], []
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        try:
            out = route(messages)
        except Exception as e:
            out = {"route": "ERR", "src": type(e).__name__, "reason": str(e), "classify_ms": 0, "wall_ms": 0}
        messages.append({"role": "assistant", "content": "ok"})
        if "expected" in turn:
            exp = turn["expected"] if isinstance(turn["expected"], list) else [turn["expected"]]
            results.append((turn["user"], exp, out))
    return results

summary = {}
for suite in args.suite:
    with open(suite if os.path.exists(suite) else os.path.join(os.path.dirname(os.path.abspath(__file__)), suite)) as f:
        cases = json.load(f)
    print(f"\n=== {suite}: {len(cases)} cases against {URL} ===")
    rows = []
    for i, case in enumerate(cases, 1):
        for user, exp, out in run_case(case, f"eval-{RUN}-{suite}-{i}"):
            ok = out["route"] in exp
            rows.append((ok, out))
            label = (case.get("name") or user)[:46].replace("\n", " ")
            note = f" ({out['reason']})" if out.get("reason") else ""
            print(f"[{'PASS' if ok else 'FAIL'}] {label!r:50} expected {'/'.join(exp):13} got {out['route']:8} via {out['src']}{note} {out['wall_ms']:.0f}ms")
        if args.gap:
            time.sleep(args.gap)
    summary[suite] = rows

print("\n================ EVAL SUMMARY ================")
all_rows = [r for rows in summary.values() for r in rows]
for suite, rows in summary.items():
    print(f"{suite}: {sum(ok for ok, _ in rows)}/{len(rows)} correct ({100 * sum(ok for ok, _ in rows) / max(len(rows), 1):.1f}%)")
print(f"TOTAL: {sum(ok for ok, _ in all_rows)}/{len(all_rows)} correct ({100 * sum(ok for ok, _ in all_rows) / max(len(all_rows), 1):.1f}%)")
srcs = {}
for _, out in all_rows:
    srcs[out["src"]] = srcs.get(out["src"], 0) + 1
print("decided by:", ", ".join(f"{k}={v}" for k, v in sorted(srcs.items())))
wall = [out["wall_ms"] for _, out in all_rows]
llm = [out["wall_ms"] for _, out in all_rows if out["src"] in ("llm", "fallback")]
print(f"route latency, all cases:  p50 {pct(wall, .5):.0f}ms  p90 {pct(wall, .9):.0f}ms")
print(f"route latency, LLM cases:  p50 {pct(llm, .5):.0f}ms  p90 {pct(llm, .9):.0f}ms  (n={len(llm)})")
sys.exit(0 if all(ok for ok, _ in all_rows) else 1)
