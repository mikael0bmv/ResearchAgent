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

Enter `exit` or press Ctrl-D to quit. Conversations are saved by default in
`.provectus/checkpoints.sqlite` (relative to the directory where you run the
command). Reuse the same thread ID and checkpoint file to continue a conversation
after restarting the CLI. You can set the conversation ID, checkpoint file, and
per-turn tool-call limit:

```sh
uv run provectus --thread-id demo --checkpoint-path .provectus/checkpoints.sqlite --max-tool-calls 2
```

If dependencies are already installed, `.venv/bin/provectus` also starts the CLI
on macOS and Linux.

## Use from Python

```python
from provectus import ResearchAgent

with ResearchAgent(max_tool_calls=2) as agent:
    reply = agent.ask("What is our remote work policy?", thread_id="demo-user")
    print(reply.answer)
```

The default SQLite checkpoint retains turns across restarts. Set
`checkpoint_path` to use another file, or pass a LangGraph `checkpointer` for a
different storage backend. Close the agent when finished, preferably with
`with ResearchAgent(...) as agent:`. Search adapters can be replaced through
`rag_search` and `web_search`; the built-in adapters return synthetic data.
External calls have a 30-second timeout by default. Set `call_timeout_seconds`
and `max_in_flight_calls` to change the latency and unfinished-call limits. A
timed-out call stops blocking the answer but may continue in a background thread;
real search adapters should also set timeouts on their own network clients.

## Test

```sh
uv run python -m unittest discover -s tests
```
