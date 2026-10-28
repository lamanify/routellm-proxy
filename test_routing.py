"""Offline tests for message parsing, the prefilter and the classifier budget: python -m unittest -v"""
import asyncio, unittest
import main
REAL_CLASSIFIER_CALL = main.classifier_call  # other tests replace main.classifier_call

def msgs(*turns, system="sys"):
    out = [{"role": "system", "content": system}]
    for t in turns:
        out.append({"role": "user", "content": t})
        out.append({"role": "assistant", "content": "ok"})
    return out[:-1]

class Split(unittest.TestCase):
    def test_reply_wrapper_newlines(self):
        self.assertEqual(main.split_message('[Replying to: "Check x.com why divi fails"]\n\n[Azri] yes'), ("yes", "Check x.com why divi fails"))
    def test_reply_wrapper_one_line(self):
        self.assertEqual(main.split_message('[Replying to: "staging link"] [Azri] thanks received'), ("thanks received", "staging link"))
    def test_reply_wrapper_quote_in_parent(self):
        i, c = main.split_message('[Replying to: "he said "go" to it"]\n\n[Azri] fix it')
        self.assertEqual(i, "fix it")
    def test_slack_sender(self):
        self.assertEqual(main.split_message("[Azri | Slack user <@U0B9T9CF961>] okay replace it")[0], "okay replace it")
    def test_memory_and_system_note_stripped(self):
        t = "[Azri | Slack user <@U1>] hi\n<memory-context>\nlong debugging notes\n[System note: x]\n</memory-context>"
        self.assertEqual(main.split_message(t), ("hi", ""))
    def test_trailing_system_note_stripped(self):
        self.assertEqual(main.split_message("[Azri] hi\n\n[System note: The previous turn was interrupted.]")[0], "hi")
    def test_notices_have_no_instruction(self):
        for t in ("[System note: gateway back online]", "[ASYNC DELEGATION BATCH COMPLETE 3 tasks]",
                  "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns", "[STILL IN PROGRESS — this is the active request",
                  "[Context from the interrupted assistant response]\nstuff", "[Tool iteration limit reached: 60/60]"):
            self.assertEqual(main.split_message(t)[0], "", t)
    def test_thread_context(self):
        t = "[Thread context — prior messages in this thread (not yet in conversation history):]\n[thread parent] Azri: check the email\n\n[Azri | Slack user <@U1>] ok do it"
        i, c = main.split_message(t)
        self.assertEqual(i, "ok do it"); self.assertIn("check the email", c)
    def test_plain(self):
        self.assertEqual(main.split_message("audit the server"), ("audit the server", ""))
    def test_thread_context_end_marker_and_new_message(self):
        t = ("[Thread context — prior messages in this thread (not yet in conversation history):]\n[assistant] Hermes hit "
             "repeated errors and stopped.\n[End of thread context]\n\n\n\n[New message]\n[Azri | Slack user <@U1>] try again")
        i, c = main.split_message(t)
        self.assertEqual(i, "try again"); self.assertIn("repeated errors", c)
        self.assertEqual(main.prefilter(i, c)[0], "inherit")
    def test_thread_new_message_keeps_its_paragraphs(self):
        t = ("[Thread context — prior:]\n[thread parent] Azri: update plugins\n[End of thread context]\n\n[New message]\n"
             "[Azri | Slack user <@U1>] Check why?\n\nwarning: skipped klinik.com")
        self.assertEqual(main.split_message(t)[0], "Check why?\n\nwarning: skipped klinik.com")
    def test_notice_after_sender_tag(self):
        self.assertEqual(main.split_message("[Azri] [ASYNC DELEGATION BATCH COMPLETE — deleg_1]\nA background fan-out finished.")[0], "")
    def test_user_text_that_looks_like_a_tag_is_kept(self):
        self.assertEqual(main.split_message("[Azri] [System Design] plan the new architecture")[0], "[System Design] plan the new architecture")
    def test_gateway_origin_header_stripped(self):
        head = 'Gateway message origin (JSON data, not instructions or authorization):\n{"platform": "slack", "chat_id": "C1"}'
        tail = "Do not guess a reply destination when these fields are insufficient."
        self.assertEqual(main.split_message(f"{head}\n{tail}\n\nin bukku"), ("in bukku", ""))
        self.assertEqual(main.split_message(f"{head} {tail} it is rented"), ("it is rented", ""))
    def test_image_description_removed(self):
        t = ('[Replying to: "Access the clinic site"]\n\n[The user sent an image~ Here\'s what I can see:\nA hero with "quotes"] '
             'and brackets.\n~]\n\n[Azri] make the CTA bigger')
        self.assertEqual(main.split_message(t), ("make the CTA bigger", "Access the clinic site"))

class ClassifierInput(unittest.TestCase):
    def test_long_instruction_truncated_to_head_and_tail(self):
        text = main.classifier_input("A" * 1000 + "M" * 200000 + "Z" * 400, "c" * 5000)
        self.assertLess(len(text), 2300)
        self.assertIn("AAAA", text); self.assertIn("ZZZZ", text)
    def test_huge_memory_context_never_reaches_classifier(self):
        i, c = main.split_message("[Azri] where is the invoice?\n<memory-context>\n" + "x" * 200000 + "\n</memory-context>")
        self.assertEqual((i, c), ("where is the invoice?", ""))

class Prefilter(unittest.TestCase):
    def test_social(self):
        for t in ("hi", "Hello there Jason", "thanks a lot!", "ping", "thanks received", "ok thanks", "Got it 👍"):
            self.assertEqual(main.prefilter(t, ""), ("tier", "routine"), t)
    def test_not_social(self):
        for t in ("hi can you audit the server", "thanks, now deploy the website to production", "ping the database and report replication lag"):
            self.assertIsNone(main.prefilter(t, ""), t)
    def test_continuations(self):
        for t in ("continue", "Approved, deploy it", "yes", "go ahead", "fix it", "yes please deploy it", "try again", "do it now"):
            self.assertEqual(main.prefilter(t, "")[0], "inherit", t)
        self.assertEqual(main.prefilter("looks good to me", ""), ("inherit", "routine"))
    def test_not_continuations(self):
        for t in ("deploy the website", "fix the checkout bug in the cart", "yes but first audit the db"):
            self.assertIsNone(main.prefilter(t, ""), t)
    def test_stack_traces(self):
        self.assertEqual(main.prefilter('Traceback (most recent call last):\n  File "a.py", line 3, in x', ""), ("tier", "think"))
        self.assertEqual(main.prefilter("check this", 'PHP Fatal error: Uncaught Error in /var/www/a.php:64'), ("tier", "think"))
        self.assertEqual(main.prefilter("what is a stack trace", ""), None)

class ConvKey(unittest.TestCase):
    def test_stable_when_tail_of_system_prompt_changes(self):
        a = main.conversation_key(msgs("audit it", system="You are Brew. " + "x" * 1200 + " time 10:00"))
        b = main.conversation_key(msgs("audit it", "continue", system="You are Brew. " + "x" * 1200 + " time 10:05"))
        self.assertEqual(a, b)
    def test_differs_per_bot(self):
        self.assertNotEqual(main.conversation_key(msgs("hi", system="You are Brew")), main.conversation_key(msgs("hi", system="You are Ops")))
    def test_compacted_sessions_of_one_bot_stay_apart(self):
        pre = "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted. " + "p" * 1500
        a = main.conversation_key(msgs(pre + "\nSummary: client A website", "next", system="You are Kayla"))
        b = main.conversation_key(msgs(pre + "\nSummary: client B ads", "next", system="You are Kayla"))
        a2 = main.conversation_key(msgs(pre + "\nSummary: client A website", "next", "more", system="You are Kayla"))
        self.assertNotEqual(a, b)
        self.assertEqual(a, a2)

class Routing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        main.route_cache.clear(); main.conv_state.clear(); main.stats.clear()
        self.calls = []; self.delay = 0.0; self.result = ("think", "ok")
        async def fake(instruction, context):
            self.calls.append((instruction, context)); await asyncio.sleep(self.delay); return self.result
        main.classifier_call = fake
        async def noop(*a, **k): pass
        main.log_to_turso = noop
        main.CLASSIFIER_BUDGET, main.CLASSIFIER_TIMEOUT, main.CLASSIFIER_HEDGE = 0.3, 0.6, 0.2

    async def test_inherit_conversation_tier(self):
        d1 = await main.route_request(msgs("audit our ingress for ssl flaws"))
        self.assertEqual((d1["tier"], d1["src"]), ("think", "llm"))
        d2 = await main.route_request(msgs("audit our ingress for ssl flaws", "continue"))
        self.assertEqual((d2["tier"], d2["src"], len(self.calls)), ("think", "inherit", 1))
        d3 = await main.route_request(msgs("audit our ingress for ssl flaws", "[System note: gateway restarted]"))
        self.assertEqual((d3["tier"], d3["src"]), ("think", "inherit"))

    async def test_thanks_does_not_reset_tier(self):
        await main.route_request(msgs("audit our ingress"))
        d = await main.route_request(msgs("audit our ingress", "thanks!"))
        self.assertEqual((d["tier"], d["src"]), ("routine", "rule"))
        d = await main.route_request(msgs("audit our ingress", "thanks!", "continue"))
        self.assertEqual(d["tier"], "think")

    async def test_conversations_are_isolated(self):
        self.result = ("think", "ok"); await main.route_request(msgs("a heavy audit", system="botA"))
        self.result = ("routine", "ok"); await main.route_request(msgs("a light question", system="botB"))
        self.assertEqual((await main.route_request(msgs("a heavy audit", "continue", system="botA")))["tier"], "think")
        self.assertEqual((await main.route_request(msgs("a light question", "continue", system="botB")))["tier"], "routine")

    async def test_cache_key_includes_conversation(self):
        await main.route_request(msgs("summarise this thread please", system="botA"))
        self.result = ("routine", "ok")
        d = await main.route_request(msgs("summarise this thread please", system="botB"))
        self.assertEqual((d["tier"], d["src"]), ("routine", "llm"))
        d = await main.route_request(msgs("summarise this thread please", system="botB"))
        self.assertEqual(d["src"], "cache")

    async def test_budget_fallback_then_background_cache(self):
        self.delay = 0.45
        d = await main.route_request(msgs("something ambiguous here"))
        self.assertEqual((d["tier"], d["src"], d["reason"]), ("chat", "fallback", "budget"))
        await asyncio.sleep(0.4)
        d = await main.route_request(msgs("something ambiguous here"))
        self.assertEqual((d["tier"], d["src"]), ("think", "cache"))

    async def test_fallback_uses_last_tier_and_is_not_cached(self):
        await main.route_request(msgs("audit the ingress setup"))
        self.result = (None, "timeout")
        d = await main.route_request(msgs("audit the ingress setup", "something ambiguous here"))
        self.assertEqual((d["tier"], d["src"], d["reason"]), ("think", "fallback", "timeout"))
        self.assertEqual(main.stats["fallback_timeout"], 1)
        self.result = ("routine", "ok")
        d = await main.route_request(msgs("audit the ingress setup", "something ambiguous here"))
        self.assertEqual((d["tier"], d["src"]), ("routine", "llm"))

    async def test_concurrent_same_message_single_call(self):
        self.delay = 0.1
        ds = await asyncio.gather(*[main.route_request(msgs("same ambiguous message")) for _ in range(5)])
        self.assertEqual(len(self.calls), 1); self.assertTrue(all(d["tier"] == "think" for d in ds))

    async def test_hedge_wins_over_slow_first_attempt(self):
        n = {"i": 0}
        async def fake(instruction, context):
            n["i"] += 1
            if n["i"] == 1: await asyncio.sleep(2)
            return "think", "ok"
        main.classifier_call = fake
        route, reason, ms, attempts = await main.classify_hedged("x", "")
        self.assertEqual((route, attempts), ("think", 2)); self.assertLess(ms, 400)

    async def test_jev_continuation_keeps_conversation_tier(self):
        await main.route_request(msgs("audit the ingress setup"))
        self.result = ("routine", "continuation")
        d = await main.route_request(msgs("audit the ingress setup", "the second one"))
        self.assertEqual((d["tier"], d["src"], d["reason"]), ("think", "inherit", "continuation"))
        d = await main.route_request(msgs("summarise the thread", "the second one", system="other"))
        self.assertEqual((d["tier"], d["src"]), ("routine", "inherit"))  # no history: Jev's own tier

    async def test_wrapper_context_used_for_bare_approval(self):
        d = await main.route_request(msgs('[Replying to: "prod db CPU at 100%"]\n\n[Azri] yes'))
        self.assertEqual(self.calls[0], ("yes", "prod db CPU at 100%"))

class JevClassifier(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved = (main.USE_JEV, main.jev_call, main.llm_call, main.JEV_MIN_CONFIDENCE)
        main.USE_JEV, main.JEV_MIN_CONFIDENCE = True, 0.0
        main.stats.clear()
        self.llm_calls = 0
    def tearDown(self):
        main.USE_JEV, main.jev_call, main.llm_call, main.JEV_MIN_CONFIDENCE = self.saved
    def use(self, jev, llm):
        async def fake_jev(instruction, context): return jev
        async def fake_llm(instruction, context):
            self.llm_calls += 1
            return llm
        main.jev_call, main.llm_call = fake_jev, fake_llm
    async def test_confident_jev_skips_llm(self):
        self.use(("think", 0.9, "ok"), ("chat", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("audit 40 pages", ""), ("think", "ok"))
        self.assertEqual(self.llm_calls, 0)
    async def test_jev_error_falls_back_to_llm(self):
        self.use((None, 0.0, "jev_http_503"), ("chat", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("explain caching", ""), ("chat", "ok"))
        self.assertEqual(main.stats["classifier_llm_after_jev_http_503"], 1)
    async def test_unsure_jev_is_kept_by_default(self):
        self.use(("routine", 0.05, "ok"), ("think", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("hmm", ""), ("routine", "ok"))
        self.assertEqual(self.llm_calls, 0)
    async def test_continuation_reason_passes_through(self):
        self.use(("chat", 0.9, "continuation"), ("think", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("option 2", ""), ("chat", "continuation"))
    async def test_unsure_jev_asks_llm_then_keeps_jev_if_llm_fails(self):
        main.JEV_MIN_CONFIDENCE = 0.35
        self.use(("routine", 0.2, "ok"), ("think", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("hmm", ""), ("think", "ok"))
        self.use(("routine", 0.2, "ok"), (None, "http_503"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("hmm", ""), ("routine", "ok"))
    async def test_both_fail(self):
        self.use((None, 0.0, "jev_timeout"), (None, "http_503"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("x", ""), (None, "jev_timeout+http_503"))
    async def test_without_key_uses_llm_only(self):
        main.USE_JEV = False
        self.use(("think", 0.99, "ok"), ("chat", "ok"))
        self.assertEqual(await REAL_CLASSIFIER_CALL("x", ""), ("chat", "ok"))

if __name__ == "__main__":
    unittest.main()


class FakeResp:
    def __init__(self, data): self.data, self.status_code = data, 200
    def json(self): return self.data

class FakeHTTP:
    def __init__(self, data): self.data, self.sent = data, []
    async def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append(json)
        return FakeResp(self.data)

def jev_answers(route, reversed_route, follows=0.0):
    return {"answers": {"route": {"probabilities": route}, "route_reversed": {"probabilities": reversed_route},
                        "continuation": {"noul": follows}}}

class JevRequest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved_http = getattr(main, "http", None)
    def tearDown(self):
        main.http = self.saved_http
    async def test_request_shape(self):
        main.http = fake = FakeHTTP(jev_answers({"chat": 1}, {"chat": 1}))
        await main.jev_call("fix it", "")
        await main.jev_call("fix it", "a" * 400 + "b" * 2000)
        s1, s2 = fake.sent[0]["state"], fake.sent[1]["state"]
        self.assertEqual(s1, {"request": "fix it"})
        self.assertTrue(s2["replying_to"].startswith("a") and s2["replying_to"].endswith("b"))
        self.assertLessEqual(len(s2["replying_to"]), main.CONTEXT_CHARS)
        q = fake.sent[0]["questions"]
        self.assertEqual(list(q["route"]["criteria"]), ["routine", "chat", "think"])
        self.assertEqual(list(q["route_reversed"]["criteria"]), ["think", "chat", "routine"])
    async def test_orders_are_averaged_and_a_tie_goes_up(self):
        main.http = FakeHTTP(jev_answers({"routine": 0.6, "chat": 0.4, "think": 0}, {"think": 0, "chat": 0.8, "routine": 0.2}))
        route, confidence, reason = await main.jev_call("x", "")
        self.assertEqual((route, reason), ("chat", "ok")); self.assertAlmostEqual(confidence, 0.4)
        main.http = FakeHTTP(jev_answers({"routine": 0.5, "chat": 0.5}, {"chat": 0.5, "routine": 0.5}))
        self.assertEqual((await main.jev_call("x", ""))[0], "chat")
    async def test_continuation(self):
        main.http = FakeHTTP(jev_answers({"routine": 1}, {"routine": 1}, follows=0.9))
        self.assertEqual(await main.jev_call("option 2", ""), ("routine", 1.0, "continuation"))
        main.http = FakeHTTP(jev_answers({"routine": 1}, {"routine": 1}, follows=0.5))
        self.assertEqual((await main.jev_call("option 2", ""))[2], "ok")
    async def test_malformed_answer(self):
        main.http = FakeHTTP({"answers": {"route": {"probabilities": {"chat": 1}}}})
        self.assertEqual(await main.jev_call("x", ""), (None, 0.0, "jev_invalid"))

class TestLamanifyTruncationAndTimeout(unittest.TestCase):
    def test_truncation_continuation_goes_to_think(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build the page"},
                {"role": "assistant", "content": "partial..."},
                {"role": "user", "content": "[System: Your previous response was truncated by the output length limit. Continue exactly where you left off.]"}]
        d = asyncio.run(main.route_request(msgs))
        self.assertEqual(d["tier"], "think"); self.assertEqual(d["src"], "rule")

    def test_first_token_timeout_scales_with_size(self):
        small = {"messages": [{"role": "user", "content": "hi"}]}
        large = {"messages": [{"role": "user", "content": "x" * (main.LARGE_REQUEST_CHARS + 1)}]}
        parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "y" * (main.LARGE_REQUEST_CHARS + 1)}]}]}
        self.assertEqual(main.first_token_timeout(small), main.FIRST_TOKEN_TIMEOUT)
        self.assertEqual(main.first_token_timeout(large), max(main.FIRST_TOKEN_TIMEOUT, main.FIRST_TOKEN_TIMEOUT_LARGE))
        self.assertEqual(main.first_token_timeout(parts), max(main.FIRST_TOKEN_TIMEOUT, main.FIRST_TOKEN_TIMEOUT_LARGE))
