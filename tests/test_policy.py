"""Policy-governed tool execution with agentbarrier."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import pytest
from agentbarrier import Decision
from agentbarrier.errors import (
    ActionInProgress,
    ActionOutcomeUnknown,
    ApprovalRejected,
    ApprovalRequired,
)
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    RuntimeReconciliation,
    SQLiteRuntimeStore,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolCallId,
    StructuredTool,
    Tool,
    tool,
)
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, field_validator

from agloom import (
    APPROVAL_INTERRUPT,
    RECONCILIATION_INTERRUPT,
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityResolutionError,
    ConfigurationError,
    EventName,
    ExecutionError,
    ExecutionPaused,
    PolicyConfigurationError,
    PolicyManager,
    PolicyViolationError,
    create_agent,
)
from agloom.policy._middleware import PolicyMiddleware
from agloom.policy._visibility import certainly_denied
from helpers import Call, EventLog, ScriptedChatModel, allow_threads, tool_call

REVIEW = PolicyRule(
    "review large refunds",
    PolicyEffect.REQUIRE_APPROVAL,
    tool="refund",
    conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
)
POLICY = RuntimePolicy(
    version="test-v1",
    rules=(
        REVIEW,
        PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund"),
        PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_*"),
        PolicyRule("no deletes", PolicyEffect.DENY, tool="delete_*"),
        PolicyRule("review exports", PolicyEffect.REQUIRE_APPROVAL, tool="export"),
        PolicyRule("allow notify", PolicyEffect.ALLOW, tool="notify"),
    ),
)

executed: list[str] = []


@tool
def refund(request_id: str, amount: float) -> str:
    """Refund an amount for a request."""
    executed.append(request_id)
    return f"refunded {amount} for {request_id}"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id} shipped"


@tool
def delete_account(account_id: str) -> str:
    """Delete an account."""
    return "deleted"


@tool
def export(target: str) -> str:
    """Export data."""
    return "exported"


@tool
def update_state(
    request_id: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Return a graph command."""
    return Command(
        update={"messages": [ToolMessage("updated", tool_call_id=tool_call_id)]}
    )


@tool
def flaky_refund(request_id: str) -> str:
    """Refund that fails downstream."""
    raise ValueError("payment provider unavailable")


@pytest.fixture(autouse=True)
def _reset_executed() -> None:
    executed.clear()


def manager(
    store: SQLiteRuntimeStore, policy: RuntimePolicy = POLICY, **options: Any
) -> PolicyManager:
    options.setdefault("consequential", {"refund": "request_id"})
    return PolicyManager(
        policy=policy,
        store=store,
        namespace="tests",
        organization_id="acme",
        requested_by="agent",
        **options,
    )


def calls(name: str, args: dict[str, Any]) -> Callable[[Call], Any]:
    def respond(call: Call) -> Any:
        if call.last.type == "tool":
            return f"final: {call.last.text}"
        return tool_call(name, args)

    return respond


def thread(name: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": name}}


class TestConstruction:
    def test_validates_arguments(self, barrier_store: SQLiteRuntimeStore) -> None:
        with pytest.raises(ConfigurationError, match="on_denied"):
            manager(barrier_store, on_denied="ignore")
        with pytest.raises(ConfigurationError, match="must not be blank"):
            manager(barrier_store, consequential={" ": "id"})
        with pytest.raises(ConfigurationError, match="must not be blank"):
            manager(barrier_store, consequential={"refund": " "})
        with pytest.raises(ConfigurationError, match="argument name or a callable"):
            manager(barrier_store, consequential={"refund": 42})
        with pytest.raises(ConfigurationError, match="namespace"):
            PolicyManager(policy=POLICY, store=barrier_store, namespace="")

    def test_loads_policies_from_files_and_mappings(
        self, barrier_store: SQLiteRuntimeStore, tmp_path: Path
    ) -> None:
        document = {
            "version": "file-v1",
            "default": "deny",
            "rules": [
                {
                    "name": "small refunds",
                    "effect": "allow",
                    "tool": "refund",
                    "conditions": [{"path": "amount", "operator": "le", "value": 10}],
                }
            ],
        }
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        from_file = PolicyManager(policy=path, store=barrier_store)
        from_text = PolicyManager(policy=str(path), store=barrier_store)
        from_mapping = PolicyManager(policy=document, store=barrier_store)
        for loaded in (from_file, from_text, from_mapping):
            assert loaded.policy.version == "file-v1"
            assert loaded.policy.evaluate("refund", {"amount": 5}).effect.value == (
                "allow"
            )

    @pytest.mark.parametrize(
        "policy",
        [{"version": 1, "rules": []}, {"version": "v", "rules": "x"}, "missing.json"],
    )
    def test_rejects_invalid_policies(
        self, barrier_store: SQLiteRuntimeStore, policy: Any
    ) -> None:
        with pytest.raises(ConfigurationError, match="invalid runtime policy"):
            PolicyManager(policy=policy, store=barrier_store)

    def test_rejects_unsupported_policy_types(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        with pytest.raises(ConfigurationError, match="must be a RuntimePolicy"):
            PolicyManager(policy=42, store=barrier_store)  # type: ignore[arg-type]

    def test_exposes_configuration(self, barrier_store: SQLiteRuntimeStore) -> None:
        policy = manager(barrier_store, on_denied="inform_model")
        assert policy.policy is POLICY
        assert policy.store is barrier_store
        assert policy.barrier.store is barrier_store
        assert dict(policy.consequential) == {"refund": "request_id"}
        assert policy.on_denied == "inform_model"
        assert policy.is_consequential("refund")
        assert not policy.is_consequential("lookup_order")
        assert policy.visible("lookup_order")
        assert not policy.visible("delete_account")


class TestDirectExecution:
    def test_non_consequential_calls_are_checked(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)
        assert policy.execute("lookup_order", {}, lambda: "ok") == "ok"
        with pytest.raises(PolicyViolationError) as denied:
            policy.execute("delete_account", {}, lambda: "never")
        assert denied.value.rule == "no deletes"
        with pytest.raises(PolicyViolationError) as default:
            policy.execute("unknown", {}, lambda: "never")
        assert default.value.rule is None
        assert "default policy effect" in str(default.value)
        with pytest.raises(
            PolicyConfigurationError, match="not declared consequential"
        ):
            policy.execute("export", {}, lambda: "never")

    def test_consequential_calls_execute_once_and_replay(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)
        runs: list[int] = []

        def operation() -> dict[str, int]:
            runs.append(1)
            return {"refunded": len(runs)}

        arguments = {"request_id": "r-1", "amount": 5}
        assert policy.execute("refund", arguments, operation) == {"refunded": 1}
        assert policy.execute("refund", arguments, operation) == {"refunded": 1}
        assert runs == [1]

    async def test_async_consequential_calls(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)

        async def operation() -> str:
            return "done"

        assert await policy.aexecute(
            "refund", {"request_id": "a", "amount": 1}, operation
        )
        assert await policy.aexecute("lookup_order", {}, operation) == "done"
        with pytest.raises(PolicyViolationError):
            await policy.aexecute("delete_account", {}, operation)

    def test_approval_is_required_then_honored(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)
        arguments = {"request_id": "big", "amount": 500}
        with pytest.raises(ApprovalRequired) as pending:
            policy.execute("refund", arguments, lambda: "paid")
        barrier_store.decide(
            pending.value.action.action_id, Decision.APPROVE, decided_by="alice"
        )
        assert policy.execute("refund", arguments, lambda: "paid") == "paid"

    def test_rejected_and_failed_actions(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)
        arguments = {"request_id": "big", "amount": 500}
        with pytest.raises(ApprovalRequired) as pending:
            policy.execute("refund", arguments, lambda: "paid")
        barrier_store.decide(
            pending.value.action.action_id, Decision.REJECT, decided_by="bob"
        )
        with pytest.raises(ApprovalRejected):
            policy.execute("refund", arguments, lambda: "paid")

        def failing() -> str:
            raise KeyError("provider down")

        small = {"request_id": "small", "amount": 1}
        with pytest.raises(ActionOutcomeUnknown) as unknown:
            policy.execute("refund", small, failing)
        assert isinstance(unknown.value.__cause__, KeyError)
        action = unknown.value.action
        assert action.idempotency_key == "small"
        assert barrier_store.get_action(action.action_id).status.value == "unknown"
        with pytest.raises(ActionOutcomeUnknown):
            policy.execute("refund", small, lambda: "retry")
        with pytest.raises(ActionOutcomeUnknown) as other:
            policy.execute("refund", {"request_id": "other", "amount": 1}, failing)
        assert other.value.action.idempotency_key == "other"

    def test_failures_keep_their_error_when_the_store_reports_no_action(
        self, barrier_store: SQLiteRuntimeStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = manager(barrier_store)
        monkeypatch.setattr(barrier_store, "list_actions", lambda **_: ())

        def failing() -> str:
            raise KeyError("provider down")

        with pytest.raises(KeyError):
            policy.execute("refund", {"request_id": "k", "amount": 1}, failing)

    async def test_async_failures_hand_over_the_unknown_action(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(barrier_store)

        async def failing() -> str:
            raise KeyError("provider down")

        async def succeeding() -> str:
            return "paid"

        arguments = {"request_id": "x", "amount": 1}
        with pytest.raises(ActionOutcomeUnknown) as unknown:
            await policy.aexecute("refund", arguments, failing)
        assert isinstance(unknown.value.__cause__, KeyError)
        assert unknown.value.action.idempotency_key == "x"
        with pytest.raises(ActionOutcomeUnknown) as retried:
            await policy.aexecute("refund", arguments, succeeding)
        assert retried.value.__cause__ is None

    @pytest.mark.parametrize(
        ("selector", "arguments", "message"),
        [
            ("request_id", {"amount": 1}, "must be a string"),
            ("request_id", {"request_id": "  ", "amount": 1}, "is blank"),
            (lambda arguments: 7, {"amount": 1}, "must return a string"),
        ],
    )
    def test_idempotency_keys_are_validated(
        self,
        barrier_store: SQLiteRuntimeStore,
        selector: Any,
        arguments: dict[str, Any],
        message: str,
    ) -> None:
        policy = manager(barrier_store, consequential={"refund": selector})
        with pytest.raises(PolicyConfigurationError, match=message):
            policy.execute("refund", arguments, lambda: "x")

    def test_callable_idempotency_keys(self, barrier_store: SQLiteRuntimeStore) -> None:
        policy = manager(
            barrier_store,
            consequential={"refund": lambda arguments: f"order-{arguments['order']}"},
        )
        assert policy.execute("refund", {"order": 7, "amount": 1}, lambda: 1) == 1
        assert policy.execute("refund", {"order": 7, "amount": 1}, lambda: 2) == 1


class TestAgentEnforcement:
    def test_allowed_tool_runs(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(calls("lookup_order", {"order_id": "7"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(barrier_store, consequential={}),
        )
        assert "order 7 shipped" in agent.invoke("Where is order 7?").text

    def test_denied_tool_fails_closed(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        events: EventLog,
    ) -> None:
        rules = (
            PolicyRule(
                "deny lookups",
                PolicyEffect.DENY,
                tool="lookup_order",
                conditions=(ArgumentCondition("order_id", ConditionOperator.EQ, "9"),),
            ),
            PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_order"),
        )
        agent = create_agent(
            model=model(calls("lookup_order", {"order_id": "9"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={},
            ),
            observers=[events],
        )
        with pytest.raises(PolicyViolationError) as denied:
            agent.invoke("Where is order 9?")
        assert denied.value.rule == "deny lookups"
        assert EventName.TOOL_CALL_DENIED in events.names

    def test_inform_model_returns_the_denial(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        rules = (
            PolicyRule(
                "deny all lookups",
                PolicyEffect.DENY,
                tool="lookup_order",
                conditions=(ArgumentCondition("order_id", ConditionOperator.EXISTS),),
            ),
            PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_order"),
        )
        agent = create_agent(
            model=model(calls("lookup_order", {"order_id": "9"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={},
                on_denied="inform_model",
            ),
        )
        assert "was not executed" in agent.invoke("Where is order 9?").text

    def test_approval_on_undeclared_tool_is_a_configuration_error(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(calls("export", {"target": "s3"})),
            tools=[export],
            pattern="react",
            policy=manager(barrier_store, consequential={}),
        )
        with pytest.raises(
            PolicyConfigurationError, match="not declared consequential"
        ):
            agent.invoke("Export the data")

    def test_certainly_denied_tools_are_hidden(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(None, "no tools needed")
        agent = create_agent(
            model=scripted,
            tools=[lookup_order, delete_account],
            pattern="react",
            policy=manager(barrier_store, consequential={}),
        )
        agent.invoke("hello")
        assert scripted.calls[0].tools == ("lookup_order",)
        assert agent.capabilities.names == ("lookup_order",)

    def test_unknown_consequential_tools_are_rejected(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        with pytest.raises(ConfigurationError, match="not configured"):
            create_agent(
                model=model(),
                tools=[lookup_order],
                policy=manager(barrier_store),
            )

    def test_approval_pauses_then_executes_exactly_once(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        events: EventLog,
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-1", "amount": 250})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
            observers=[events],
        )
        config = thread("t-approve")
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Refund 250 for r-1", config)
        assert paused.value.thread_id == "t-approve"
        payload = paused.value.interrupts[0].value
        assert payload["type"] == APPROVAL_INTERRUPT
        assert payload["tool"] == "refund"
        assert payload["arguments"] == {"request_id": "r-1", "amount": 250}
        assert payload["policy_rule"] == "review large refunds"
        assert executed == []

        with pytest.raises(ExecutionPaused):
            agent.resume(config)
        assert executed == []

        barrier_store.decide(payload["action_id"], Decision.APPROVE, decided_by="alice")
        answer = agent.resume(config)
        assert "refunded 250" in answer.text
        assert executed == ["r-1"]
        assert events.names.count(EventName.APPROVAL_REQUIRED) == 2
        assert EventName.ACTION_EXECUTED in events.names

    def test_rejection_is_reported(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-2", "amount": 900})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store, on_denied="inform_model"),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = thread("t-reject")
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Refund 900 for r-2", config)
        action_id = paused.value.interrupts[0].value["action_id"]
        barrier_store.decide(action_id, Decision.REJECT, decided_by="bob")
        answer = agent.resume(config)
        assert "was not executed" in answer.text
        assert executed == []

    def test_completed_actions_are_replayed(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        events: EventLog,
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-3", "amount": 10})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store),
            observers=[events],
        )
        first = agent.invoke("Refund 10 for r-3")
        second = agent.invoke("Refund 10 for r-3 again")
        assert "refunded 10" in first.text
        assert "refunded 10" in second.text
        assert executed == ["r-3"]
        assert EventName.ACTION_REPLAYED in events.names

    def test_without_checkpointer_approval_required_propagates(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-4", "amount": 300})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store),
        )
        with pytest.raises(ApprovalRequired):
            agent.invoke("Refund 300 for r-4")

    async def test_async_approval_flow(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-5", "amount": 700})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = thread("t-async")
        with pytest.raises(ExecutionPaused) as paused:
            await agent.ainvoke("Refund 700 for r-5", config)
        barrier_store.decide(
            paused.value.interrupts[0].value["action_id"],
            Decision.APPROVE,
            decided_by="carol",
        )
        answer = await agent.aresume(config)
        assert "refunded 700" in answer.text
        assert executed == ["r-5"]

    async def test_async_denials_and_failures(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        denied = create_agent(
            model=model(calls("delete_account", {"account_id": "1"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(barrier_store, consequential={}, on_denied="inform_model"),
        )
        assert "final" in (await denied.ainvoke("Delete account 1")).text

        rules = (PolicyRule("allow", PolicyEffect.ALLOW, tool="flaky_refund"),)
        failing = create_agent(
            model=model(calls("flaky_refund", {"request_id": "f-1"})),
            tools=[flaky_refund],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={"flaky_refund": "request_id"},
            ),
        )
        with pytest.raises(ActionOutcomeUnknown) as error:
            await failing.ainvoke("Refund f-1")
        assert isinstance(error.value.__cause__, ValueError)
        with pytest.raises(ActionOutcomeUnknown) as retried:
            await failing.ainvoke("Refund f-1 again")
        assert retried.value.action.action_id == error.value.action.action_id

    def test_consequential_failures_are_reported_and_fail_closed(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        events: EventLog,
    ) -> None:
        rules = (PolicyRule("allow", PolicyEffect.ALLOW, tool="flaky_refund"),)
        agent = create_agent(
            model=model(calls("flaky_refund", {"request_id": "f-2"})),
            tools=[flaky_refund],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={"flaky_refund": "request_id"},
            ),
            observers=[events],
        )
        with pytest.raises(ActionOutcomeUnknown) as unknown:
            agent.invoke("Refund f-2")
        failed = next(e for e in events if e.name == EventName.ACTION_FAILED)
        assert failed.payload == {
            "tool": "flaky_refund",
            "error": "ActionOutcomeUnknown",
            "action_id": unknown.value.action.action_id,
        }
        with pytest.raises(ActionOutcomeUnknown):
            agent.invoke("Refund f-2 again")

    @pytest.mark.parametrize(
        ("reconciled", "seen"),
        [
            (
                {"content": "Refund RF-9 issued", "status": "success"},
                "Refund RF-9 issued",
            ),
            ("Refund RF-9 issued", "Refund RF-9 issued"),
            ({"refund_id": "RF-9"}, '{"refund_id": "RF-9"}'),
        ],
    )
    def test_reconciled_results_are_replayed_to_the_model(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        reconciled: Any,
        seen: str,
    ) -> None:
        rules = (PolicyRule("allow", PolicyEffect.ALLOW, tool="flaky_refund"),)
        agent = create_agent(
            model=model(calls("flaky_refund", {"request_id": "f-3"})),
            tools=[flaky_refund],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={"flaky_refund": "request_id"},
            ),
        )
        with pytest.raises(ActionOutcomeUnknown) as unknown:
            agent.invoke("Refund f-3")
        barrier_store.reconcile(
            unknown.value.action.action_id,
            RuntimeReconciliation.COMMITTED,
            resolved_by="operator",
            reason="confirmed with the payment provider",
            result=reconciled,
        )
        assert agent.invoke("Refund f-3 again").text == f"final: {seen}"

    def test_consequential_tools_cannot_return_commands(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        rules = (PolicyRule("allow", PolicyEffect.ALLOW, tool="update_state"),)
        agent = create_agent(
            model=model(calls("update_state", {"request_id": "c-1"})),
            tools=[update_state],
            pattern="react",
            policy=manager(
                barrier_store,
                RuntimePolicy(version="v", rules=rules),
                consequential={"update_state": "request_id"},
            ),
        )
        with pytest.raises(PolicyConfigurationError, match="graph Command"):
            agent.invoke("Update")


class TestReconciliation:
    """Unknown outcomes pause checkpointed runs until an operator reconciles them."""

    WIRES = RuntimePolicy(
        version="wires-v1",
        rules=(PolicyRule("allow wires", PolicyEffect.ALLOW, tool="send_wire"),),
    )

    @staticmethod
    def wire_tool(attempts: list[str], failures: set[str]) -> Any:
        @tool
        def send_wire(transfer_id: str) -> str:
            """Send a bank wire."""
            attempts.append(transfer_id)
            if transfer_id in failures:
                failures.discard(transfer_id)
                raise TimeoutError("the bank did not confirm the wire in time")
            return f"wire {transfer_id} sent"

        return send_wire

    def agent(
        self,
        store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        send_wire: Any,
        transfer_id: str,
        events: EventLog,
    ) -> Any:
        return create_agent(
            model=model(calls("send_wire", {"transfer_id": transfer_id})),
            tools=[send_wire],
            pattern="react",
            policy=manager(
                store, self.WIRES, consequential={"send_wire": "transfer_id"}
            ),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
            observers=[events],
        )

    def test_committed_outcomes_reach_the_model_on_resume(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        events: EventLog,
    ) -> None:
        attempts: list[str] = []
        send_wire = self.wire_tool(attempts, {"TX-1"})
        agent = self.agent(barrier_store, model, saver, send_wire, "TX-1", events)
        config = thread("t-wire-1")
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Wire TX-1", config)
        payload = paused.value.interrupts[0].value
        assert payload["type"] == RECONCILIATION_INTERRUPT
        assert payload["tool"] == "send_wire"
        assert payload["arguments"] == {"transfer_id": "TX-1"}
        assert payload["error"] == "TimeoutError"
        failed = [e for e in events if e.name == EventName.ACTION_FAILED]
        assert [e.payload["action_id"] for e in failed] == [payload["action_id"]]

        with pytest.raises(ExecutionPaused) as again:
            agent.resume(config)
        assert again.value.interrupts[0].value["action_id"] == payload["action_id"]
        assert attempts == ["TX-1"]
        assert events.names.count(EventName.ACTION_FAILED) == 1

        barrier_store.reconcile(
            payload["action_id"],
            RuntimeReconciliation.COMMITTED,
            resolved_by="ops",
            reason="the bank statement lists the wire",
            result="Wire TX-1 confirmed by the bank statement.",
        )
        answer = agent.resume(config)
        assert answer.text == "final: Wire TX-1 confirmed by the bank statement."
        assert attempts == ["TX-1"]
        assert EventName.ACTION_REPLAYED in events.names

    async def test_actions_not_committed_run_exactly_once_on_resume(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        events: EventLog,
    ) -> None:
        attempts: list[str] = []
        send_wire = self.wire_tool(attempts, {"TX-2"})
        agent = self.agent(barrier_store, model, saver, send_wire, "TX-2", events)
        config = thread("t-wire-2")
        with pytest.raises(ExecutionPaused) as paused:
            await agent.ainvoke("Wire TX-2", config)
        action_id = paused.value.interrupts[0].value["action_id"]
        with pytest.raises(ExecutionPaused):
            await agent.aresume(config)
        assert attempts == ["TX-2"]
        barrier_store.reconcile(
            action_id,
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by="ops",
            reason="the bank has no record of the wire",
        )
        answer = await agent.aresume(config)
        assert answer.text == "final: wire TX-2 sent"
        assert attempts == ["TX-2", "TX-2"]
        assert EventName.ACTION_EXECUTED in events.names


@tool
def payout(payout_id: str, amount: float = 5000.0) -> str:
    """Pay out an amount."""
    executed.append(payout_id)
    return f"paid {amount}"


@tool
def wipe(disk: str, force: bool = False) -> str:
    """Wipe a disk."""
    executed.append(disk)
    return "wiped"


TRANSFER_SCHEMA = {
    "type": "object",
    "properties": {
        "transfer_id": {"type": "string"},
        "amount": {"type": "number", "default": 5000},
    },
    "required": ["transfer_id"],
    "additionalProperties": False,
}


def json_transfer(transfer_id: str, amount: float) -> str:
    executed.append(transfer_id)
    return f"transferred {amount}"


@tool
def ping() -> str:
    """Check the payment service."""
    return "pong"


@tool
def annotate(note: Any) -> str:
    """Attach a note to the case."""
    return "annotated"


def store_blob(payload: Any) -> str:
    return "stored"


BLOB = StructuredTool.from_function(
    store_blob,
    name="store_blob",
    description="Store a blob.",
    args_schema={
        "type": "object",
        "properties": {"payload": {}},
        "required": ["payload"],
    },
)


class Opaque:
    """A value without a JSON form."""


class OpaqueArguments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: Opaque


def use_handle(handle: Opaque) -> str:
    return "used"


def untyped_tool() -> Tool:
    """A single-input tool without an argument schema."""
    return Tool(name="legacy", func=lambda text: text, description="Run an operation.")


class NormalizedRefund(BaseModel):
    """Refund arguments normalized and checked by the tool's own model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    request_id: str
    amount: float

    @field_validator("request_id")
    @classmethod
    def canonical(cls, value: str) -> str:
        return value.upper()

    @field_validator("amount")
    @classmethod
    def positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be positive")
        return value


def normalized_refund(request_id: str, amount: float) -> str:
    executed.append(request_id)
    return f"refunded {amount} for {request_id}"


NORMALIZED_REFUND = StructuredTool.from_function(
    normalized_refund,
    name="refund",
    description="Refund an amount for a request.",
    args_schema=NormalizedRefund,
)


class Charge(BaseModel):
    """A charge in dollars that the tool's own model converts to cents."""

    order_id: str
    amount: int

    @field_validator("amount")
    @classmethod
    def to_cents(cls, value: int) -> int:
        return value * 100


def charge_order(order_id: str, amount: int) -> str:
    executed.append(f"{order_id}:{amount}")
    return f"charged {amount} cents"


CHARGE = StructuredTool.from_function(
    charge_order,
    name="charge",
    description="Charge an order in dollars.",
    args_schema=Charge,
)


class OrderLookup(BaseTool):
    """A tool whose argument schema is inferred from `_run`."""

    name: str = "find_order"
    description: str = "Find an order."

    def _run(self, order_id: str, quantity: int = 1) -> str:
        return f"order {order_id} x{quantity!r}"


class ForeignSchemaTool(BaseTool):
    """A tool whose schema class is not a Pydantic v2 model, like a Pydantic v1 one."""

    name: str = "foreign"
    description: str = "Run an operation."
    args_schema: Any = Opaque

    def _run(self, **kwargs: Any) -> str:
        return "ran"


class VariadicLookup(BaseTool):
    """A tool whose `**kwargs` take arguments that no schema describes."""

    name: str = "variadic"
    description: str = "Look something up."

    def _run(self, query: str, **kwargs: Any) -> str:
        return query


class ConfiguredLookup(BaseTool):
    """A tool whose inferred schema asks the model for LangChain's config."""

    name: str = "configured"
    description: str = "Look something up."

    def _run(self, query: str, config: RunnableConfig) -> str:
        return query


class TestArgumentValidation:
    """The policy judges the values a tool actually receives."""

    RULES = RuntimePolicy(
        version="arguments-v1",
        rules=(
            REVIEW,
            PolicyRule(
                "review large payouts",
                PolicyEffect.REQUIRE_APPROVAL,
                tool="payout",
                conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
            ),
            PolicyRule(
                "review large transfers",
                PolicyEffect.REQUIRE_APPROVAL,
                tool="transfer",
                conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
            ),
            PolicyRule(
                "no forced wipes",
                PolicyEffect.DENY,
                tool="wipe",
                conditions=(ArgumentCondition("force", ConditionOperator.EQ, True),),
            ),
            PolicyRule("allow the rest", PolicyEffect.ALLOW, tool="*"),
        ),
    )

    def governed(
        self,
        store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        responder: Callable[[Call], Any],
        tools: list[Any],
        **options: Any,
    ) -> Any:
        keys = {
            "refund": "request_id",
            "payout": "payout_id",
            "transfer": "transfer_id",
            "charge": "order_id",
        }
        names = {item.name for item in tools}
        return create_agent(
            model=model(responder),
            tools=tools,
            pattern="react",
            policy=manager(
                store,
                self.RULES,
                consequential={
                    name: key for name, key in keys.items() if name in names
                },
                on_denied="inform_model",
            ),
            **options,
        )

    def test_converted_values_meet_approval_rules(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
    ) -> None:
        agent = self.governed(
            barrier_store,
            model,
            calls("refund", {"request_id": "r-9", "amount": "5000"}),
            [refund],
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Refund 5000", thread("t-convert"))
        request = paused.value.interrupts[0].value
        assert request["arguments"] == {"request_id": "r-9", "amount": 5000.0}
        assert executed == []

    def test_defaults_meet_approval_rules(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = self.governed(
            barrier_store, model, calls("payout", {"payout_id": "p-1"}), [payout]
        )
        with pytest.raises(ApprovalRequired) as pending:
            agent.invoke("Pay out p-1")
        assert dict(pending.value.action.arguments) == {
            "payout_id": "p-1",
            "amount": 5000.0,
        }
        assert executed == []

    def test_converted_values_meet_deny_rules(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = self.governed(
            barrier_store,
            model,
            calls("wipe", {"disk": "prod", "force": "true"}),
            [wipe],
        )
        assert "was not executed" in agent.invoke("Wipe prod").text
        assert executed == []

    def test_invalid_calls_return_to_the_model_without_an_action(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        events: EventLog,
    ) -> None:
        attempts = iter(
            [
                tool_call("refund", {"request_id": "r-5", "amount": "fifty"}),
                tool_call("refund", {"request_id": " ", "amount": 50}),
                tool_call("refund", {"request_id": "r-5", "amount": 50}),
            ]
        )

        def respond(call: Call) -> Any:
            if call.last.type == "tool" and call.last.status == "success":
                return f"final: {call.last.text}"
            return next(attempts)

        agent = self.governed(
            barrier_store, model, respond, [refund], observers=[events]
        )
        assert agent.invoke("Refund 50 for r-5").text == "final: refunded 50.0 for r-5"
        assert executed == ["r-5"]
        assert [action.idempotency_key for action in barrier_store.list_actions()] == [
            "r-5"
        ]
        denied = [e for e in events if e.name == EventName.TOOL_CALL_DENIED]
        assert [e.payload["reason"] for e in denied] == ["InvalidArguments"] * 2

    @pytest.mark.parametrize(
        ("arguments", "outcome"),
        [
            ({"transfer_id": "t-1", "amount": "5000"}, "invalid"),
            ({"transfer_id": "t-1"}, "approval"),
            ({"transfer_id": "t-1", "amount": 50}, "executed"),
        ],
    )
    def test_json_schema_tools_are_validated_strictly(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        arguments: dict[str, Any],
        outcome: str,
    ) -> None:
        transfer = StructuredTool.from_function(
            json_transfer,
            name="transfer",
            description="Transfer money.",
            args_schema=TRANSFER_SCHEMA,
        )
        agent = self.governed(
            barrier_store, model, calls("transfer", arguments), [transfer]
        )
        if outcome == "approval":
            with pytest.raises(ApprovalRequired):
                agent.invoke("Transfer")
            assert executed == []
            return
        answer = agent.invoke("Transfer").text
        if outcome == "invalid":
            assert "Invalid arguments" in answer
            assert executed == []
        else:
            assert answer == "final: transferred 50"
            assert executed == ["t-1"]

    def test_direct_execution_validates_tool_arguments(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = self.governed(barrier_store, model, calls("refund", {}), [refund])
        with pytest.raises(ApprovalRequired):
            agent.execute_capability("refund", {"request_id": "r-7", "amount": "500"})
        with pytest.raises(CapabilityResolutionError, match="invalid arguments"):
            agent.execute_capability("refund", {"request_id": "r-7", "amount": "x"})
        assert agent.execute_capability(
            "refund", {"request_id": "r-8", "amount": "50"}
        ) == ("refunded 50.0 for r-8")

    @pytest.mark.parametrize(
        ("selector", "message"),
        [("order_id", "must be a string argument"), ("amount", "string argument")],
    )
    def test_idempotency_keys_must_name_string_arguments(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        selector: str,
        message: str,
    ) -> None:
        with pytest.raises(ConfigurationError, match=message):
            create_agent(
                model=model(),
                tools=[refund],
                policy=manager(barrier_store, consequential={"refund": selector}),
            )

    @pytest.mark.parametrize(
        ("key", "accepted"),
        [
            ({"type": ["string", "null"]}, True),
            ({"$ref": "#/$defs/TransferId"}, True),
            ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, True),
            (True, True),
            ({"type": "integer"}, False),
            ({"oneOf": [{"type": "integer"}, {"type": "number"}]}, False),
            (False, False),
        ],
    )
    def test_idempotency_keys_may_be_any_string_typed_argument(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        key: Any,
        accepted: bool,
    ) -> None:
        transfer = StructuredTool.from_function(
            json_transfer,
            name="transfer",
            description="Transfer money.",
            args_schema={"type": "object", "properties": {"transfer_id": key}},
        )

        def build() -> Any:
            return create_agent(
                model=model(),
                tools=[transfer],
                policy=manager(
                    barrier_store,
                    self.RULES,
                    consequential={"transfer": "transfer_id"},
                ),
            )

        if accepted:
            assert "transfer" in build().capabilities.names
        else:
            with pytest.raises(ConfigurationError, match="string argument"):
                build()

    @pytest.mark.parametrize(
        ("governed_tool", "message"),
        [
            (untyped_tool(), "variadic or RunnableConfig parameters"),
            (VariadicLookup(), "variadic or RunnableConfig parameters"),
            (ConfiguredLookup(), "variadic or RunnableConfig parameters"),
            (ForeignSchemaTool(), "not a Pydantic v2 model or a JSON schema"),
            (
                StructuredTool.from_function(
                    use_handle,
                    name="use_handle",
                    description="Use a handle.",
                    args_schema=OpaqueArguments,
                ),
                "no JSON form",
            ),
        ],
    )
    def test_governed_tools_need_a_schema_the_policy_can_judge(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        governed_tool: BaseTool,
        message: str,
    ) -> None:
        policy = manager(barrier_store, self.RULES, consequential={})
        with pytest.raises(ConfigurationError, match=message):
            create_agent(model=model(), tools=[governed_tool], policy=policy)
        with pytest.raises(ConfigurationError, match=message):
            create_agent(
                model=model(),
                capabilities={governed_tool.name: governed_tool},
                policy=policy,
            )

    def test_direct_execution_accepts_every_tool_input_shape(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(),
            tools=[lookup_order, ping, annotate, BLOB],
            policy=manager(barrier_store, self.RULES, consequential={}),
        )
        assert agent.execute_capability("lookup_order", "o-1") == "order o-1 shipped"
        assert agent.execute_capability("ping") == "pong"
        with pytest.raises(CapabilityResolutionError, match="must be an object"):
            agent.execute_capability("lookup_order", ["o-1"])
        with pytest.raises(CapabilityResolutionError, match="not JSON-compatible"):
            agent.execute_capability("annotate", {"note": object()})
        with pytest.raises(CapabilityResolutionError, match="not JSON-compatible"):
            agent.execute_capability("store_blob", {"payload": object()})
        # Capabilities registered after creation are validated when they run.
        agent.capabilities.register(
            CapabilityDescriptor(
                name="legacy",
                description="Run an operation.",
                kind=CapabilityKind.APPLICATION,
                value=untyped_tool(),
            )
        )
        with pytest.raises(CapabilityResolutionError, match="declares no argument"):
            agent.execute_capability("legacy", {})

    def test_tool_validators_shape_the_judged_arguments(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        def run(request_id: str) -> str:
            agent = self.governed(
                barrier_store,
                model,
                calls("refund", {"request_id": request_id, "amount": 5}),
                [NORMALIZED_REFUND],
            )
            return agent.invoke("Refund 5").text

        assert run("r-1") == "final: refunded 5.0 for R-1"
        assert run(" R-1 ") == "final: refunded 5.0 for R-1"
        assert executed == ["R-1"]
        assert [action.idempotency_key for action in barrier_store.list_actions()] == [
            "R-1"
        ]

    def test_validator_rejections_never_claim_the_action(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        attempts = iter(
            [
                tool_call("refund", {"request_id": "r-2", "amount": -5}),
                tool_call("refund", {"request_id": "r-2", "amount": 5}),
            ]
        )

        def respond(call: Call) -> Any:
            if call.last.type == "tool" and call.last.status == "success":
                return f"final: {call.last.text}"
            return next(attempts)

        agent = self.governed(barrier_store, model, respond, [NORMALIZED_REFUND])
        assert agent.invoke("Refund 5 for r-2").text == "final: refunded 5.0 for R-2"
        assert executed == ["R-2"]
        assert [action.idempotency_key for action in barrier_store.list_actions()] == [
            "R-2"
        ]

    def test_tool_validators_run_once(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = self.governed(
            barrier_store,
            model,
            calls("charge", {"order_id": "o-1", "amount": 5}),
            [CHARGE],
        )
        assert agent.invoke("Charge 5 dollars").text == "final: charged 500 cents"
        assert agent.execute_capability("charge", {"order_id": "o-2", "amount": 7}) == (
            "charged 700 cents"
        )
        assert executed == ["o-1:500", "o-2:700"]
        judged = sorted(
            dict(action.arguments)["amount"] for action in barrier_store.list_actions()
        )
        assert judged == [500, 700]

    def test_inferred_schemas_pass_on_the_judged_values(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = self.governed(
            barrier_store,
            model,
            calls("find_order", {"order_id": "o-1", "quantity": "3"}),
            [OrderLookup()],
        )
        assert agent.invoke("Find o-1").text == "final: order o-1 x3"
        assert agent.execute_capability("find_order", {"order_id": "o-2"}) == (
            "order o-2 x1"
        )

    async def test_async_calls_are_validated(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        attempts = iter(
            [
                tool_call("refund", {"request_id": "r-5", "amount": "fifty"}),
                tool_call("refund", {"request_id": "r-5", "amount": 50}),
            ]
        )

        def respond(call: Call) -> Any:
            if call.last.type == "tool" and call.last.status == "success":
                return f"final: {call.last.text}"
            return next(attempts)

        agent = self.governed(barrier_store, model, respond, [refund])
        answer = await agent.ainvoke("Refund 50 for r-5")
        assert answer.text == "final: refunded 50.0 for r-5"
        assert await agent.aexecute_capability(
            "refund", {"request_id": "r-6", "amount": "40"}
        ) == ("refunded 40.0 for r-6")
        assert executed == ["r-5", "r-6"]

    def test_callable_idempotency_keys_govern_agent_calls(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(calls("refund", {"request_id": "r-3", "amount": 20})),
            tools=[refund],
            pattern="react",
            policy=manager(
                barrier_store,
                self.RULES,
                consequential={
                    "refund": lambda arguments: f"refund:{arguments['request_id']}"
                },
            ),
        )
        assert agent.invoke("Refund 20").text == "final: refunded 20.0 for r-3"
        assert [action.idempotency_key for action in barrier_store.list_actions()] == [
            "refund:r-3"
        ]


class TestConcurrentDuplicates:
    @staticmethod
    def duplicate_wires() -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "send_wire", "args": {"transfer_id": "TX-9"}, "id": "c1"},
                {"name": "send_wire", "args": {"transfer_id": "TX-9"}, "id": "c2"},
            ],
        )

    def agent(
        self,
        store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        send_wire: Any,
    ) -> Any:
        def respond(call: Call) -> Any:
            results = [m.text for m in call.messages if m.type == "tool"]
            return f"final: {results}" if results else self.duplicate_wires()

        return create_agent(
            model=model(respond),
            tools=[send_wire],
            pattern="react",
            policy=manager(
                store,
                TestReconciliation.WIRES,
                consequential={"send_wire": "transfer_id"},
            ),
        )

    def test_duplicates_wait_for_the_running_call_and_replay_it(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        sent: list[str] = []

        @tool
        def send_wire(transfer_id: str) -> str:
            """Send a bank wire."""
            time.sleep(0.3)
            sent.append(transfer_id)
            return f"wire {transfer_id} sent"

        answer = self.agent(barrier_store, model, send_wire).invoke("Wire TX-9")
        assert sent == ["TX-9"]
        assert answer.text == "final: ['wire TX-9 sent', 'wire TX-9 sent']"

    async def test_async_duplicates_wait_for_the_running_call(
        self, barrier_store: SQLiteRuntimeStore, model: Callable[..., ScriptedChatModel]
    ) -> None:
        sent: list[str] = []

        @tool
        async def send_wire(transfer_id: str) -> str:
            """Send a bank wire."""
            await asyncio.sleep(0.3)
            sent.append(transfer_id)
            return f"wire {transfer_id} sent"

        answer = await self.agent(barrier_store, model, send_wire).ainvoke("Wire TX-9")
        assert sent == ["TX-9"]
        assert answer.text == "final: ['wire TX-9 sent', 'wire TX-9 sent']"

    def test_direct_duplicates_report_the_running_action(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        policy = manager(
            barrier_store,
            TestReconciliation.WIRES,
            consequential={"send_wire": "transfer_id"},
        )
        started, release = threading.Event(), threading.Event()

        def send() -> str:
            started.set()
            release.wait(5)
            return "wire TX-9 sent"

        wire = {"transfer_id": "TX-9"}
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(policy.execute, "send_wire", wire, send)
            assert started.wait(5)
            # Applications decide how to answer a request already in progress.
            with pytest.raises(ActionInProgress):
                policy.execute("send_wire", wire, send)
            release.set()
            assert first.result(5) == "wire TX-9 sent"
        assert policy.execute("send_wire", wire, send) == "wire TX-9 sent"

    def test_policy_middleware_requires_an_agloom_run(
        self, barrier_store: SQLiteRuntimeStore
    ) -> None:
        middleware = PolicyMiddleware(manager(barrier_store))
        request = ToolCallRequest(
            tool_call={"name": "refund", "args": {}, "id": "c1"},
            tool=None,
            state={},
            runtime=None,
        )
        with pytest.raises(ExecutionError, match="only inside Agloom agents"):
            middleware.wrap_tool_call(request, lambda _: ToolMessage("", "c1"))


tool_names = st.sampled_from(["refund", "refund_v2", "lookup", "delete_all", "x"])
patterns = st.sampled_from(["*", "refund", "refund*", "lookup", "delete_*", "?"])
paths = st.sampled_from(["amount", "region", "nested.flag"])
scalars = st.one_of(st.integers(-5, 500), st.sampled_from(["eu", "us"]), st.booleans())
Op = ConditionOperator
condition = st.one_of(
    st.builds(ArgumentCondition, paths, st.just(Op.EXISTS), st.booleans()),
    st.builds(ArgumentCondition, paths, st.sampled_from([Op.EQ, Op.NE]), scalars),
    st.builds(
        ArgumentCondition,
        paths,
        st.sampled_from([Op.LT, Op.LE, Op.GT, Op.GE]),
        st.integers(-5, 500),
    ),
    st.builds(
        ArgumentCondition,
        paths,
        st.sampled_from([Op.IN, Op.NOT_IN]),
        st.lists(scalars, max_size=3),
    ),
    st.builds(
        ArgumentCondition,
        paths,
        st.sampled_from([Op.CONTAINS, Op.STARTS_WITH, Op.ENDS_WITH]),
        st.sampled_from(["e", "u", "eu"]),
    ),
)
rules = st.lists(
    st.tuples(
        st.sampled_from(list(PolicyEffect)),
        patterns,
        st.lists(condition, max_size=2).map(tuple),
    ),
    max_size=5,
).map(
    lambda parts: tuple(
        PolicyRule(f"rule-{index}", effect, tool=pattern, conditions=conditions)
        for index, (effect, pattern, conditions) in enumerate(parts)
    )
)
arguments = st.fixed_dictionaries(
    {},
    optional={
        "amount": st.integers(-10, 1_000),
        "region": st.sampled_from(["eu", "us", "apac"]),
        "nested": st.fixed_dictionaries({"flag": st.booleans()}),
    },
)


@given(rules, st.sampled_from(list(PolicyEffect)), tool_names, arguments)
def test_hidden_tools_are_denied_for_every_call(
    policy_rules: tuple[PolicyRule, ...],
    default: PolicyEffect,
    name: str,
    call_arguments: dict[str, Any],
) -> None:
    policy = RuntimePolicy(version="h", rules=policy_rules, default_effect=default)
    if certainly_denied(policy, name):
        assert policy.evaluate(name, call_arguments).effect is PolicyEffect.DENY


class ApprovalMachine(RuleBasedStateMachine):
    """Approvals never execute early, at most once, and never after rejection."""

    pending = Bundle("pending")

    @initialize()
    def setup(self) -> None:
        import tempfile

        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = SQLiteRuntimeStore(str(Path(self.folder.name) / "machine.db"))
        self.policy = manager(self.store)
        self.runs: dict[str, int] = {}
        self.approved: set[str] = set()
        self.rejected: set[str] = set()
        self.counter = 0

    def teardown(self) -> None:
        close = getattr(getattr(self, "store", None), "close", None)
        if callable(close):
            close()
        folder = getattr(self, "folder", None)
        if folder is not None:
            folder.cleanup()

    def _call(self, key: str) -> Any:
        def operation() -> str:
            self.runs[key] = self.runs.get(key, 0) + 1
            return key

        return self.policy.execute(
            "refund", {"request_id": key, "amount": 500}, operation
        )

    @rule(target=pending)
    def request(self) -> str:
        self.counter += 1
        key = f"k{self.counter}"
        with pytest.raises(ApprovalRequired):
            self._call(key)
        return key

    @rule(key=pending)
    def approve(self, key: str) -> None:
        if key in self.approved or key in self.rejected:
            return
        action = self._action(key)
        self.store.decide(action, Decision.APPROVE, decided_by="alice")
        self.approved.add(key)

    @rule(key=pending)
    def reject(self, key: str) -> None:
        if key in self.approved or key in self.rejected:
            return
        self.store.decide(self._action(key), Decision.REJECT, decided_by="bob")
        self.rejected.add(key)

    @rule(key=pending)
    def retry(self, key: str) -> None:
        if key in self.approved:
            assert self._call(key) == key
        elif key in self.rejected:
            with pytest.raises(ApprovalRejected):
                self._call(key)
        else:
            with pytest.raises(ApprovalRequired):
                self._call(key)

    def _action(self, key: str) -> str:
        (action,) = (
            action
            for action in self.store.list_actions()
            if action.idempotency_key == key
        )
        return action.action_id

    @invariant()
    def executes_at_most_once_and_only_when_approved(self) -> None:
        for key, count in getattr(self, "runs", {}).items():
            assert count == 1
            assert key in self.approved
            assert key not in self.rejected


ApprovalMachine.TestCase.settings = settings(
    max_examples=25, stateful_step_count=12, deadline=None
)
TestApprovalMachine = ApprovalMachine.TestCase


class TestAsyncEnforcement:
    async def test_async_allow_deny_and_reject(
        self,
        barrier_store: SQLiteRuntimeStore,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
    ) -> None:
        rules = (
            PolicyRule(
                "deny order 9",
                PolicyEffect.DENY,
                tool="lookup_order",
                conditions=(ArgumentCondition("order_id", ConditionOperator.EQ, "9"),),
            ),
            PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_order"),
        )
        policy = RuntimePolicy(version="v", rules=rules)
        allowed = create_agent(
            model=model(calls("lookup_order", {"order_id": "7"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(barrier_store, policy, consequential={}),
        )
        assert "order 7 shipped" in (await allowed.ainvoke("Where is 7?")).text
        denied = create_agent(
            model=model(calls("lookup_order", {"order_id": "9"})),
            tools=[lookup_order],
            pattern="react",
            policy=manager(barrier_store, policy, consequential={}),
        )
        with pytest.raises(PolicyViolationError):
            await denied.ainvoke("Where is 9?")

        rejecting = create_agent(
            model=model(calls("refund", {"request_id": "async-r", "amount": 900})),
            tools=[refund],
            pattern="react",
            policy=manager(barrier_store, on_denied="inform_model"),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = thread("async-reject")
        with pytest.raises(ExecutionPaused) as paused:
            await rejecting.ainvoke("Refund 900", config)
        barrier_store.decide(
            paused.value.interrupts[0].value["action_id"],
            Decision.REJECT,
            decided_by="bob",
        )
        assert "was not executed" in (await rejecting.aresume(config)).text
        assert executed == []
