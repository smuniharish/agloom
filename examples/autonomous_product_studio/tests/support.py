"""Deterministic doubles for the Product Studio tests."""

from __future__ import annotations

import itertools
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from agentbarrier.runtime import SQLiteRuntimeStore
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from procrastinate.testing import InMemoryConnector

from backend.governance import default_policy
from backend.jobs import JobQueue, job_app
from backend.services import StudioServices
from backend.settings import StudioSettings
from backend.store import MemoryStudioStore
from backend.streaming import RunEventBroker

_IDS = itertools.count(1)
PREVIEW_URL = "http://target-preview"


class ScriptedModel(GenericFakeChatModel):
    """LangChain's fake chat model that also accepts tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        del tools, kwargs
        return self


def scripted(*responses: AIMessage | str) -> ScriptedModel:
    return ScriptedModel(messages=iter(responses))


def tool_call(tool: str, /, **arguments: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool, "args": arguments, "id": f"call_{next(_IDS)}"}],
    )


class RecordingBoundary:
    """An execution boundary that records checks instead of running them."""

    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.calls: list[list[tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict[str, object]:
        del workspace, timeout
        with self._lock:
            self.calls.append(list(commands))
        return {
            "passed": self.passed,
            "exit_code": 0 if self.passed else 1,
            "output": "",
        }


class KeywordEmbedder:
    def __init__(self, vocabulary: Iterable[str] = ("checkout", "weather")) -> None:
        self.vocabulary = tuple(vocabulary)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(word in text.lower()) for word in self.vocabulary] + [0.1]
            for text in texts
        ]


def memory_settings(tmp_path: Path, **changes: Any) -> StudioSettings:
    values: dict[str, Any] = {
        "database_url": "memory://",
        "projects_root": tmp_path / "projects",
        "preview_url": PREVIEW_URL,
        "browser_validation": False,
        "embedded_worker": True,
        "worker_poll_interval": 0.05,
        "shutdown_grace_seconds": 5,
    }
    values.update(changes)
    return StudioSettings(**values)


def memory_services(
    tmp_path: Path,
    model: Any,
    *,
    store: MemoryStudioStore | None = None,
    boundary: Any = None,
    **settings: Any,
) -> Callable[[], Any]:
    """A services factory wired like production, with in-memory backends."""
    studio_store = store or MemoryStudioStore()

    @asynccontextmanager
    async def factory() -> AsyncIterator[StudioServices]:
        policy_store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
        try:
            with job_app.replace_connector(InMemoryConnector()):
                async with job_app.open_async():
                    yield StudioServices(
                        settings=memory_settings(tmp_path, **settings),
                        store=studio_store,
                        events=RunEventBroker(studio_store, poll_interval=0.05),
                        jobs=JobQueue(job_app),
                        policy=default_policy(PREVIEW_URL),
                        policy_store=policy_store,
                        checkpointer=InMemorySaver(),
                        model_factory=lambda: model,
                        execution_boundary=boundary,
                    )
        finally:
            policy_store.close()

    return factory
