"""Provectus research assistant."""

from .agent import AgentExecutionError, AgentReply, ResearchAgent, mock_rag_search, mock_web_search

__all__ = ["AgentExecutionError", "AgentReply", "ResearchAgent", "mock_rag_search", "mock_web_search", "main"]


def main() -> None:
    """Run a local, mocked conversation from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Mock LangGraph research assistant")
    parser.add_argument("--thread-id", default="local-demo")
    parser.add_argument("--max-tool-calls", type=int, default=2)
    args = parser.parse_args()
    agent = ResearchAgent(max_tool_calls=args.max_tool_calls)
    print("Mock research assistant. Press Ctrl-D or enter 'exit' to quit.")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        try:
            print(f"Assistant: {agent.ask(query, thread_id=args.thread_id).answer}")
        except AgentExecutionError as exc:
            print(f"Assistant: {exc}")
