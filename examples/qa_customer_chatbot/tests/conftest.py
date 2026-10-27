from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from backend.store import Store
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted replies, accepts tool binding, and records its inputs."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def scripted_model() -> Callable[..., ScriptedModel]:
    def build(*replies: AIMessage) -> ScriptedModel:
        return ScriptedModel(messages=iter(replies))

    return build
