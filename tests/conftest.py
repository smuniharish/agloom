"""Shared fixtures and Hypothesis profiles."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from agentbarrier.runtime import SQLiteRuntimeStore
from hypothesis import HealthCheck, settings
from langgraph.checkpoint.memory import InMemorySaver

from helpers import EventLog, Responder, ScriptedChatModel

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=400,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def model() -> Callable[..., ScriptedChatModel]:
    """Return a factory for scripted chat models."""

    def build(responder: Responder | None = None, *queue: Any) -> ScriptedChatModel:
        return ScriptedChatModel(responder=responder, queue=list(queue))

    return build


@pytest.fixture
def saver() -> InMemorySaver:
    """Return an in-memory LangGraph checkpointer."""
    return InMemorySaver()


@pytest.fixture
def barrier_store(tmp_path: Path) -> Iterator[SQLiteRuntimeStore]:
    """Return an agentbarrier SQLite store in a temporary directory."""
    store = SQLiteRuntimeStore(str(tmp_path / "barrier.db"))
    yield store
    store.close()


@pytest.fixture
def events() -> EventLog:
    """Return an event recorder usable as an observer."""
    return EventLog()
