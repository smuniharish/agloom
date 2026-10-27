"""Context carry-over: each iteration sees the previous result, within bounds.

The next iteration receives the previous result, evaluator feedback, and
errors as untrusted context (data, never instructions), and
`LoopPolicy(max_context_chars=...)` caps how much of the previous result is
carried. This example records the prompts the model actually receives
(through an observer's LangChain callback handler) and shows that the second
pass of a two-pass draft-and-polish loop sees the bounded first draft.

Run:
    uv run python examples/38_loop_context.py
"""

from typing import Any

from _support import real_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agloom import (
    AgentEvent,
    DefaultLoopEvaluator,
    LoopPolicy,
    LoopTerminationReason,
    Observer,
    create_agent,
)

CONTEXT_CHARS = 300


class ModelCalls(BaseCallbackHandler):
    """Records the prompt and the answer of every model call."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.answers: list[str] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self.prompts.append("\n".join(message.text for message in messages[0]))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.answers.append(response.generations[0][0].text)


class PromptAudit(Observer):
    """Attaches `ModelCalls` to every model call of the agent."""

    def __init__(self) -> None:
        self.calls = ModelCalls()

    def on_event(self, event: AgentEvent) -> None:
        return

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        return (self.calls,)


def main() -> None:
    audit = PromptAudit()
    agent = create_agent(
        model=real_model(),
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=2, max_context_chars=CONTEXT_CHARS),
        loop_evaluator=DefaultLoopEvaluator(),
        observers=[audit],
    )

    result = agent.invoke_loop(
        "Draft a 120-word announcement of Agloom's new policy engine for the "
        "engineering blog, then polish it on the next pass."
    )

    first_draft, second_prompt = audit.calls.answers[0], audit.calls.prompts[1]
    assert result.state.termination_reason is LoopTerminationReason.ITERATION_LIMIT
    assert len(first_draft) > CONTEXT_CHARS, len(first_draft)
    assert first_draft[:CONTEXT_CHARS] in second_prompt
    assert first_draft not in second_prompt
    print(f"Pass 2 received the first {CONTEXT_CHARS} of {len(first_draft)} chars.")
    print(result.result.text)


if __name__ == "__main__":
    main()
