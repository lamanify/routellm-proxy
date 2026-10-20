"""Offline tests for message parsing, the prefilter and the classifier budget: python -m unittest -v"""
import asyncio, unittest
import main

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

    async def test_wrapper_context_used_for_bare_approval(self):
        d = await main.route_request(msgs('[Replying to: "prod db CPU at 100%"]\n\n[Azri] yes'))
        self.assertEqual(self.calls[0], ("yes", "prod db CPU at 100%"))

if __name__ == "__main__":
    unittest.main()
