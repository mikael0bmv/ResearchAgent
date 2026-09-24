import unittest

from langgraph.checkpoint.memory import InMemorySaver

from provectus import ResearchAgent


class ResearchAgentTests(unittest.TestCase):
    def test_internal_and_external_routing_with_citations(self):
        calls = []

        def rag(query):
            calls.append("rag")
            return [{"id": "I1", "title": "Internal", "location": "mock://internal",
                     "content": "Internal evidence", "kind": "internal"}]

        def web(query):
            calls.append("web")
            return [{"id": "W1", "title": "Web", "location": "mock://web",
                     "content": "External evidence", "kind": "web"}]

        agent = ResearchAgent(rag_search=rag, web_search=web)
        reply = agent.ask("Compare our internal policy with current public guidance", thread_id="a")
        self.assertEqual(calls, ["rag", "web"])
        self.assertEqual(reply.tool_calls, 2)
        self.assertIn("[I1]", reply.answer)
        self.assertIn("[W1]", reply.answer)

    def test_circuit_breaker_and_turn_reset(self):
        calls = []

        def search(query):
            calls.append(query)
            return []

        agent = ResearchAgent(rag_search=search, web_search=search, max_tool_calls=1)
        first = agent.ask("Our policy and current public guidance", thread_id="a")
        self.assertEqual(first.tool_calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("Tool call limit reached", first.answer)
        second = agent.ask("What about onboarding?", thread_id="a")
        self.assertEqual(second.tool_calls, 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("Tool call limit reached", second.answer)

    def test_checkpoint_keeps_history_and_isolates_threads(self):
        agent = ResearchAgent()
        agent.ask("Tell me about remote work", thread_id="a")
        agent.ask("What about it?", thread_id="a")
        state_a = agent.graph.get_state({"configurable": {"thread_id": "a"}}).values
        self.assertEqual(len(state_a["messages"]), 4)
        self.assertIn("remote work", state_a["search_query"].lower())
        agent.ask("Latest public news", thread_id="b")
        state_b = agent.graph.get_state({"configurable": {"thread_id": "b"}}).values
        self.assertEqual(len(state_b["messages"]), 2)
        self.assertEqual(state_b["sources"][0]["kind"], "web")

    def test_injected_checkpointer_restores_history_in_new_agent(self):
        checkpointer = InMemorySaver()
        ResearchAgent(checkpointer=checkpointer).ask("Latest public news", thread_id="a")
        agent = ResearchAgent(checkpointer=checkpointer)
        agent.ask("What about it?", thread_id="a")
        state = agent.graph.get_state({"configurable": {"thread_id": "a"}}).values
        self.assertEqual(len(state["messages"]), 4)
        self.assertTrue(state["needs_web"])
        self.assertFalse(state["needs_rag"])

    def test_search_failure_is_reported_and_other_tool_continues(self):
        def failing_search(query):
            raise TimeoutError("private detail")

        agent = ResearchAgent(rag_search=failing_search)
        reply = agent.ask("Our policy and current public guidance", thread_id="a")
        self.assertEqual(reply.tool_calls, 2)
        self.assertEqual(reply.sources[0]["kind"], "web")
        self.assertIn("Internal search is unavailable", reply.answer)
        self.assertNotIn("private detail", reply.answer)

    def test_follow_up_reuses_prior_route_and_search_context(self):
        calls = []

        def web(query):
            calls.append(("web", query))
            return []

        def rag(query):
            calls.append(("rag", query))
            return []

        agent = ResearchAgent(rag_search=rag, web_search=web)
        agent.ask("Latest public news about Acme", thread_id="a")
        agent.ask("What about it?", thread_id="a")
        self.assertEqual([name for name, _ in calls], ["web", "web"])
        self.assertIn("Acme", calls[-1][1])

        agent.ask("Tell me about our remote work policy", thread_id="b")
        combined = agent.ask("How does that compare with current public guidance?", thread_id="b")
        self.assertEqual([name for name, _ in calls[-3:]], ["rag", "rag", "web"])
        self.assertEqual(combined.tool_calls, 2)
        self.assertIn("remote work policy", calls[-1][1])

    def test_invalid_source_is_rejected_without_leaking_content(self):
        def bad_rag(query):
            return [{"id": "I1", "title": "Bad", "location": "mock://bad",
                     "content": "untrusted result", "kind": "web"}]

        agent = ResearchAgent(rag_search=bad_rag)
        reply = agent.ask("Our policy", thread_id="a")
        self.assertEqual(reply.sources, [])
        self.assertEqual(reply.tool_calls, 1)
        self.assertIn("Internal search is unavailable", reply.answer)
        self.assertNotIn("untrusted result", reply.answer)

    def test_zero_budget_skips_tools(self):
        calls = []

        def search(query):
            calls.append(query)
            return []

        agent = ResearchAgent(rag_search=search, web_search=search, max_tool_calls=0)
        reply = agent.ask("Our policy and current news", thread_id="a")
        self.assertEqual(calls, [])
        self.assertEqual(reply.tool_calls, 0)
        self.assertIn("Tool call limit reached", reply.answer)

    def test_invalid_input_is_rejected_before_graph_execution(self):
        agent = ResearchAgent()
        with self.assertRaises(ValueError):
            agent.ask("   ", thread_id="a")
        with self.assertRaises(ValueError):
            agent.ask("Question", thread_id=" ")
        with self.assertRaises(ValueError):
            ResearchAgent(max_tool_calls=True)


if __name__ == "__main__":
    unittest.main()
