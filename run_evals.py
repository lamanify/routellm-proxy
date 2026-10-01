import os, json, time, urllib.request

PROXY_URL = os.environ.get("ROUTER_URL", "https://routellm.assistant.lamanify.com/v1").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

with open("eval_suite.json", "r") as f:
    cases = json.load(f)

print(f"Running evaluation on {len(cases)} cases against {PROXY_URL}...")
passed = 0
results = []

for i, c in enumerate(cases, 1):
    prompt = c["prompt"]
    expected = c["expected"]
    req = urllib.request.Request(
        f"{PROXY_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5
        }).encode()
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            chosen = resp.headers.get("X-RouteLLM-Choice", "unknown")
            dur = round((time.time() - t0) * 1000, 1)
            is_match = (chosen == expected)
            if is_match:
                passed += 1
            print(f"[{i:02d}/{len(cases)}] {prompt[:40]!r:45} -> Expected: {expected:7} | Got: {chosen:7} [{PASS if is_match else FAIL}] ({dur}ms)")
    except Exception as e:
        print(f"[{i:02d}/{len(cases)}] {prompt[:40]!r:45} -> ERR: {e}")

accuracy = round((passed / len(cases)) * 100, 2)
print(f"\n================ EVAL SUMMARY ================")
print(f"Total: {len(cases)} | Passed: {passed} | Accuracy: {accuracy}%")
print(f"==============================================")
