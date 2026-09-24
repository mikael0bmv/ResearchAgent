# Provectus

A LangGraph research assistant that searches mock internal documents, optionally
searches mock web results, and answers with source citations. No search service or
API key is needed to run the demo.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Run the CLI

From the project directory:

```sh
uv sync
uv run provectus
```

At the `You:` prompt, enter a question. For example:

```text
You: What is our remote work policy?
Assistant: In this demo, remote work requests go through a manager. [I1]

Sources (mock data):
[I1] Demo remote work policy (mock://internal/remote-work)
```

Enter `exit` or press Ctrl-D to quit. The CLI keeps conversation context during
the session. You can set the conversation ID and per-turn tool-call limit:

```sh
uv run provectus --thread-id demo --max-tool-calls 2
```

If dependencies are already installed, `.venv/bin/provectus` also starts the CLI
on macOS and Linux.

## Use from Python

```python
from provectus import ResearchAgent

agent = ResearchAgent(max_tool_calls=2)
reply = agent.ask("What is our remote work policy?", thread_id="demo-user")
print(reply.answer)
```

The default `InMemorySaver` checkpoint keeps turns while the process runs. To
retain conversations across restarts, pass a durable LangGraph checkpointer to
`ResearchAgent(checkpointer=...)`. Search adapters can be replaced through
`rag_search` and `web_search`; the built-in adapters return synthetic data.

## Test

```sh
uv run python -m unittest discover -s tests
```
