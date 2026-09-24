"""A small, testable LangGraph research assistant with durable checkpoints."""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class Source(TypedDict):
    id: str
    title: str
    location: str
    content: str
    kind: Literal["internal", "web"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    search_query: str
    needs_rag: bool
    needs_web: bool
    sources: list[Source]
    errors: list[str]
    tool_calls: int


SearchTool = Callable[[str], list[Source]]


def mock_rag_search(query: str) -> list[Source]:
    """Return illustrative internal documents; no external service is called."""
    documents: list[Source] = [
        {"id": "I1", "title": "Demo remote work policy", "location": "mock://internal/remote-work",
         "content": "In this demo, remote work requests go through a manager.", "kind": "internal"},
        {"id": "I2", "title": "Demo onboarding guide", "location": "mock://internal/onboarding",
         "content": "In this demo, new hires receive an onboarding checklist.", "kind": "internal"},
    ]
    terms = set(re.findall(r"[a-z]{4,}", query.lower())) - {"what", "when", "where", "which", "about", "with", "from", "does", "your", "this", "that", "internal", "policy"}
    return [doc for doc in documents if terms & set(re.findall(r"[a-z]{4,}", (doc["title"] + " " + doc["content"]).lower()))]


def mock_web_search(query: str) -> list[Source]:
    """A visibly synthetic web result used to exercise external routing."""
    return [{"id": "W1", "title": "Mock external search result", "location": "mock://web/search",
             "content": f"This is a simulated search result for: {query}. It contains no verified public facts.",
             "kind": "web"}]


@dataclass(frozen=True)
class AgentReply:
    answer: str
    sources: list[Source]
    errors: list[str]
    tool_calls: int


class AgentExecutionError(RuntimeError):
    """The workflow could not complete or persist a turn."""


class ResearchAgent:
    def __init__(
        self,
        *,
        rag_search: SearchTool = mock_rag_search,
        web_search: SearchTool = mock_web_search,
        model: BaseChatModel | None = None,
        checkpointer=None,
        checkpoint_path: str | Path = ".provectus/checkpoints.sqlite",
        max_tool_calls: int = 2,
        call_timeout_seconds: float = 30.0,
        max_in_flight_calls: int = 8,
    ) -> None:
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a nonnegative integer")
        if (isinstance(call_timeout_seconds, bool) or not isinstance(call_timeout_seconds, (int, float))
                or not math.isfinite(call_timeout_seconds) or call_timeout_seconds <= 0):
            raise ValueError("call_timeout_seconds must be a positive finite number")
        if (isinstance(max_in_flight_calls, bool) or not isinstance(max_in_flight_calls, int)
                or max_in_flight_calls < 1):
            raise ValueError("max_in_flight_calls must be a positive integer")
        self.rag_search = rag_search
        self.web_search = web_search
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.call_timeout_seconds = float(call_timeout_seconds)
        self._call_slots = BoundedSemaphore(max_in_flight_calls)
        self._checkpoint_connection: sqlite3.Connection | None = None
        if checkpointer is None:
            try:
                path = Path(checkpoint_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._checkpoint_connection = sqlite3.connect(path, timeout=5, check_same_thread=False)
                checkpointer = SqliteSaver(self._checkpoint_connection)
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                if self._checkpoint_connection is not None:
                    self._checkpoint_connection.close()
                    self._checkpoint_connection = None
                raise AgentExecutionError("The checkpoint database could not be opened.") from exc
        self.checkpointer = checkpointer

        graph = StateGraph(AgentState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("rag", self._rag)
        graph.add_node("web", self._web)
        graph.add_node("synthesize", self._synthesize)
        graph.add_edge(START, "analyze")
        graph.add_conditional_edges("analyze", self._next_after_analyze,
                                    {"rag": "rag", "web": "web", "synthesize": "synthesize"})
        graph.add_conditional_edges("rag", self._next_after_rag,
                                    {"web": "web", "synthesize": "synthesize"})
        graph.add_edge("web", "synthesize")
        graph.add_edge("synthesize", END)
        try:
            self.graph = graph.compile(checkpointer=self.checkpointer)
        except Exception as exc:
            self.close()
            raise AgentExecutionError("The research graph could not be initialized.") from exc

    def close(self) -> None:
        """Close the SQLite connection owned by this agent, if any."""
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def __enter__(self) -> ResearchAgent:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def ask(self, query: str, *, thread_id: str) -> AgentReply:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        try:
            state = self.graph.invoke(
                {"messages": [HumanMessage(content=query.strip())]},
                config={"configurable": {"thread_id": thread_id}},
            )
            messages = state["messages"]
            if not messages or not isinstance(messages[-1], AIMessage):
                raise ValueError("research graph returned no assistant answer")
            return AgentReply(
                answer=str(messages[-1].content),
                sources=state.get("sources", []),
                errors=state.get("errors", []),
                tool_calls=state.get("tool_calls", 0),
            )
        except Exception as exc:
            logger.exception("Research graph failed for thread %s", thread_id)
            raise AgentExecutionError("The research request could not be completed.") from exc

    def _analyze(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        query = next((str(msg.content) for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")
        lowered = query.lower()
        web_terms = ("latest", "current", "today", "recent", "online", "web", "public", "external", "news")
        internal_terms = ("internal", "company", "our ", "policy", "onboarding", "document", "docs", "remote work")
        needs_web = any(term in lowered for term in web_terms)
        needs_rag = any(term in lowered for term in internal_terms)
        search_query = query
        # Resolve references against the last turn and retain its search route.
        words = set(re.findall(r"[a-z]+", lowered))
        if words & {"it", "that", "this", "those", "they", "them", "their"}:
            prior = [str(msg.content) for msg in messages[:-1] if isinstance(msg, HumanMessage)]
            if prior:
                search_query = f"{prior[-1]} {query}"
                needs_rag = needs_rag or state.get("needs_rag", False)
                needs_web = needs_web or state.get("needs_web", False)
        if not needs_rag and not needs_web:
            needs_rag = True
        return {"query": query, "search_query": search_query, "needs_rag": needs_rag,
                "needs_web": needs_web, "sources": [], "errors": [], "tool_calls": 0}

    def _next_after_analyze(self, state: AgentState) -> str:
        if state["tool_calls"] >= self.max_tool_calls:
            return "synthesize"
        if state["needs_rag"]:
            return "rag"
        if state["needs_web"]:
            return "web"
        return "synthesize"

    def _next_after_rag(self, state: AgentState) -> str:
        if state["needs_web"] and state["tool_calls"] < self.max_tool_calls:
            return "web"
        return "synthesize"

    def _invoke_bounded(self, call: Callable, *args):
        """Bound external-call latency and the number of unfinished calls."""
        if not self._call_slots.acquire(blocking=False):
            raise RuntimeError("external call capacity reached")
        result: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def run() -> None:
            try:
                result.put((True, call(*args)))
            except BaseException as exc:
                result.put((False, exc))
            finally:
                self._call_slots.release()

        try:
            Thread(target=run, name="provectus-external-call", daemon=True).start()
        except Exception:
            self._call_slots.release()
            raise
        try:
            succeeded, value = result.get(timeout=self.call_timeout_seconds)
        except Empty as exc:
            raise TimeoutError("external call timed out") from exc
        if not succeeded:
            if isinstance(value, Exception):
                raise value
            raise RuntimeError("external call failed")
        return value

    def _run_search(self, state: AgentState, name: str, kind: Literal["internal", "web"], search: SearchTool) -> AgentState:
        if state["tool_calls"] >= self.max_tool_calls:
            return {"errors": [*state["errors"], "Tool call limit reached."], "tool_calls": state["tool_calls"]}
        errors = list(state["errors"])
        sources = list(state["sources"])
        try:
            results = self._invoke_bounded(search, state["search_query"])
            required = ("id", "title", "location", "content", "kind")
            if not isinstance(results, list) or any(
                not isinstance(item, dict)
                or any(not isinstance(item.get(key), str) or not item[key].strip() for key in required)
                or item["kind"] != kind
                or not re.fullmatch(r"[A-Z]\d+", item["id"])
                for item in results
            ) or len({item["id"] for item in results} | {item["id"] for item in sources}) != len(results) + len(sources):
                raise ValueError("search returned invalid sources")
            sources.extend(results)
        except Exception:
            logger.exception("%s search failed", name)
            errors.append(f"{name} search is unavailable.")
        return {"sources": sources, "errors": errors, "tool_calls": state["tool_calls"] + 1}

    def _rag(self, state: AgentState) -> AgentState:
        return self._run_search(state, "Internal", "internal", self.rag_search)

    def _web(self, state: AgentState) -> AgentState:
        return self._run_search(state, "Web", "web", self.web_search)

    @staticmethod
    def _validate_extractive_answer(draft: str, sources: list[Source]) -> set[str]:
        """Accept only source text copied verbatim with a matching citation."""
        source_by_id = {source["id"]: source for source in sources}
        lines = [line.strip() for line in draft.splitlines() if line.strip()]
        if not lines:
            raise ValueError("model returned no cited excerpts")
        used: set[str] = set()
        for line in lines:
            match = re.fullmatch(r"(.+?)\s+\[([A-Z]\d+)\]", line)
            if match is None:
                raise ValueError("model answer contains an uncited line")
            excerpt, source_id = match.groups()
            source = source_by_id.get(source_id)
            if source is None or excerpt not in source["content"]:
                raise ValueError("model answer contains unsupported text")
            used.add(source_id)
        return used

    def _synthesize(self, state: AgentState) -> AgentState:
        sources = state["sources"]
        errors = list(state["errors"])
        requested = int(state["needs_rag"]) + int(state["needs_web"])
        if state["tool_calls"] < requested:
            errors.append("Tool call limit reached; some searches were skipped.")
        if sources:
            lines = [f"{source['content']} [{source['id']}]" for source in sources]
            answer = "\n".join(lines)
            answer += "\n\nSources (mock data):\n" + "\n".join(
                f"[{source['id']}] {source['title']} ({source['location']})" for source in sources
            )
        else:
            answer = "I found no supporting sources for this query."
        if self.model is not None and sources:
            try:
                context = "\n".join(f"[{s['id']}] {s['content']}" for s in sources)
                history = state["messages"][-6:]
                response = self._invoke_bounded(self.model.invoke, [
                    SystemMessage(content="Select relevant text verbatim from the supplied mock sources. "
                                          "Write one excerpt per line, followed by its source ID in square "
                                          "brackets. Do not paraphrase or add other prose."),
                    *history,
                    HumanMessage(content=f"Question: {state['query']}\nSources:\n{context}"),
                ])
                draft = str(response.content).strip()
                used = self._validate_extractive_answer(draft, sources)
                answer = draft + "\n\nSources (mock data):\n" + "\n".join(
                    f"[{s['id']}] {s['title']} ({s['location']})" for s in sources if s["id"] in used
                )
            except Exception:
                logger.exception("Answer synthesis failed; using source excerpts")
                errors.append("Answer synthesis was unavailable; source excerpts are shown.")
        if errors:
            answer += "\n\nLimitations: " + " ".join(errors)
        return {"messages": [AIMessage(content=answer)], "errors": errors}
