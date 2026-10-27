"""Policy-governed tool execution built on agentbarrier."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from agentbarrier.errors import (
    ActionInProgress,
    ActionLimitExceeded,
    ActionLimitValueError,
    ActionOutcomeUnknown,
    ApprovalExpired,
    ApprovalRejected,
    ApprovalRequired,
    EmergencyPauseActive,
    PolicyDenied,
)
from agentbarrier.runtime import (
    IdempotencySelector,
    PolicyEffect,
    RuntimeAction,
    RuntimeBarrier,
    RuntimeObserver,
    RuntimePolicy,
    RuntimeStatus,
    RuntimeStore,
)
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.types import Command, interrupt

from agloom._runtime import RunContext
from agloom.errors import (
    AgloomError,
    ConfigurationError,
    PolicyConfigurationError,
    PolicyViolationError,
)
from agloom.observability.events import EventName
from agloom.policy._arguments import InvalidArguments, call_arguments
from agloom.policy._visibility import certainly_denied

DeniedBehavior = Literal["raise", "inform_model"]
APPROVAL_INTERRUPT = "agloom.approval"
RECONCILIATION_INTERRUPT = "agloom.reconciliation"

_PRE_EXECUTION_DENIALS = (
    PolicyDenied,
    ApprovalRejected,
    ApprovalExpired,
    EmergencyPauseActive,
    ActionLimitExceeded,
    ActionLimitValueError,
)
_DEFAULT_RULE = "<default>"
_FIRST_WAIT_SECONDS = 0.05
_MAX_WAIT_SECONDS = 1.0

ToolHandler = Callable[[ToolCallRequest], ToolMessage | Command[Any]]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


@dataclass
class _Attempt:
    """Tracks whether a consequential operation ran and whether it finished."""

    started: bool = False
    executed: bool = False

    def wrap(self, operation: Callable[[], Any]) -> Callable[[], Any]:
        def tracked() -> Any:
            self.started = True
            result = operation()
            self.executed = True
            return result

        return tracked

    def awrap(
        self, operation: Callable[[], Awaitable[Any]]
    ) -> Callable[[], Awaitable[Any]]:
        async def tracked() -> Any:
            self.started = True
            result = await operation()
            self.executed = True
            return result

        return tracked

    def raised(self, error: Exception) -> bool:
        """Whether the operation itself failed with an application error."""
        return self.started and not self.executed and not isinstance(error, AgloomError)


class PolicyManager:
    """Governs every tool call of an agent with an agentbarrier runtime policy.

    Applications supply their own agentbarrier `RuntimePolicy` (as an object,
    a strict JSON policy file, or a mapping), a durable `RuntimeStore`, and the
    service identity the actions are bound to.

    * **Consequential tools** (declared in `consequential` with a business
      idempotency key) are durably mediated by agentbarrier: approval binding,
      exactly-once execution, replay of stored results, emergency pauses,
      action limits, and audit receipts.
    * **Every other tool** is checked statelessly with the same policy:
      `allow` runs the tool and `deny` raises
      [`PolicyViolationError`][agloom.PolicyViolationError]. A rule that
      requires approval for an undeclared tool fails closed with
      [`PolicyConfigurationError`][agloom.PolicyConfigurationError].
    * Tools the policy denies for every possible argument are hidden from the model.
    * Every model tool call is validated against the tool's argument schema
      first, so rules judge the values the tool receives. An invalid call
      never runs: the model receives an error tool message and can correct it.

    A duplicate model call made while the same consequential action executes
    waits for that execution and replays its result.

    When the agent has a checkpointer, a pending approval pauses the run with a
    LangGraph interrupt; resuming after a reviewer decided executes the approved
    call exactly once. Without a checkpointer, agentbarrier's `ApprovalRequired`
    propagates to the caller.

    When a consequential tool raises, its outcome is unknown, and agentbarrier
    refuses to run the action again until an operator reconciles it. With a
    checkpointer, the run pauses with a reconciliation interrupt; resuming after
    reconciliation retries the call, which replays the operator's result when
    the action was committed and runs exactly once when it was not. Without a
    checkpointer, and for direct `execute` calls, agentbarrier's
    `ActionOutcomeUnknown` is raised with the action, chained from the tool's
    error. Results reconciled as committed reach the model as their `content`,
    or as JSON text when they have another shape.

    Args:
        policy: The policy, a path to a strict JSON policy file, or a mapping.
        store: Durable agentbarrier store (SQLite, PostgreSQL, or custom).
        consequential: Tool name to idempotency selector: the name of a
            string argument or a callable deriving the business key from the
            canonical arguments.
        namespace: Service namespace the actions are bound to.
        organization_id: Tenant the actions are bound to.
        requested_by: Requesting service identity.
        on_denied: `"raise"` (fail closed) or `"inform_model"`, which returns
            pre-execution denials to the model as error tool messages.
        observer: Optional agentbarrier runtime observer, for example an
            OpenTelemetry observer.

    Raises:
        ConfigurationError: If the policy cannot be loaded or an argument is invalid.
    """

    def __init__(
        self,
        *,
        policy: RuntimePolicy | str | os.PathLike[str] | Mapping[str, Any],
        store: RuntimeStore,
        consequential: Mapping[str, IdempotencySelector] | None = None,
        namespace: str = "default",
        organization_id: str = "default",
        requested_by: str | None = None,
        on_denied: DeniedBehavior = "raise",
        observer: RuntimeObserver | None = None,
    ) -> None:
        if on_denied not in ("raise", "inform_model"):
            raise ConfigurationError("on_denied must be 'raise' or 'inform_model'")
        self._policy = _load_policy(policy)
        resolved: dict[str, IdempotencySelector] = {}
        for name, selector in (consequential or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ConfigurationError("consequential tool names must not be blank")
            if isinstance(selector, str) and not selector.strip():
                raise ConfigurationError(
                    f"idempotency key of {name!r} must not be blank"
                )
            if not isinstance(selector, str) and not callable(selector):
                raise ConfigurationError(
                    f"idempotency key of {name!r} must be an argument name "
                    "or a callable"
                )
            resolved[name] = selector
        self._consequential = MappingProxyType(resolved)
        self._on_denied: DeniedBehavior = on_denied
        try:
            self._barrier = RuntimeBarrier(
                policy=self._policy,
                store=store,
                namespace=namespace,
                organization_id=organization_id,
                requested_by=requested_by,
                observer=observer,
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error

    @property
    def policy(self) -> RuntimePolicy:
        """The enforced runtime policy."""
        return self._policy

    @property
    def store(self) -> RuntimeStore:
        """The durable agentbarrier store used for consequential tools."""
        return self._barrier.store

    @property
    def barrier(self) -> RuntimeBarrier:
        """The agentbarrier runtime barrier that mediates consequential tools."""
        return self._barrier

    @property
    def consequential(self) -> Mapping[str, IdempotencySelector]:
        """The declared consequential tools and their idempotency selectors."""
        return self._consequential

    @property
    def on_denied(self) -> DeniedBehavior:
        """How pre-execution denials of model tool calls are reported."""
        return self._on_denied

    def visible(self, tool_name: str) -> bool:
        """Return whether some call of the tool could be allowed or approved."""
        return not certainly_denied(self._policy, tool_name)

    def is_consequential(self, tool_name: str) -> bool:
        """Return whether the tool is durably mediated."""
        return tool_name in self._consequential

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        operation: Callable[[], Any],
    ) -> Any:
        """Run an application operation under the policy, outside a tool call.

        The operation gets the same rules, approvals, and exactly-once
        guarantees as an agent tool with that name. Consequential results must
        be JSON-compatible because agentbarrier stores them for replay.
        Approvals never pause here, and a call made while the same action is
        executing is refused rather than awaited: for consequential calls,
        agentbarrier's errors propagate unchanged, for example `PolicyDenied`,
        `ApprovalRequired`, `ActionInProgress`, and `ActionOutcomeUnknown`
        (chained from the operation's error when the operation raised).

        Raises:
            PolicyViolationError: If the policy denies a non-consequential call.
            PolicyConfigurationError: If an approval rule matches a
                non-consequential call, or no idempotency key can be derived
                from the arguments.
        """
        return self._execute(tool_name, arguments, operation, None)

    async def aexecute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Asynchronously run an application operation under the policy.

        See `execute`.

        Raises:
            PolicyViolationError: If the policy denies a non-consequential call.
            PolicyConfigurationError: If an approval rule matches a
                non-consequential call, or no idempotency key can be derived
                from the arguments.
        """
        return await self._aexecute(tool_name, arguments, operation, None)

    def _execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        operation: Callable[[], Any],
        run: RunContext | None,
    ) -> Any:
        if tool_name not in self._consequential:
            self._check(tool_name, arguments, run)
            return operation()
        key = _select_key(self._consequential[tool_name], arguments, tool_name)
        attempt = _Attempt()
        try:
            result = self._barrier.execute(
                tool_name=tool_name,
                arguments=arguments,
                idempotency_key=key,
                operation=attempt.wrap(operation),
            )
        except Exception as error:
            failure = self._failure(tool_name, key, error, attempt)
            _report_failure(run, tool_name, failure)
            if failure is error:
                raise
            raise failure from error
        _report_success(run, tool_name, executed=attempt.executed)
        return result

    async def _aexecute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        operation: Callable[[], Awaitable[Any]],
        run: RunContext | None,
    ) -> Any:
        if tool_name not in self._consequential:
            self._check(tool_name, arguments, run)
            return await operation()
        key = _select_key(self._consequential[tool_name], arguments, tool_name)
        attempt = _Attempt()
        try:
            result = await self._barrier.execute_async(
                tool_name=tool_name,
                arguments=arguments,
                idempotency_key=key,
                operation=attempt.awrap(operation),
            )
        except Exception as error:
            failure = self._failure(tool_name, key, error, attempt)
            _report_failure(run, tool_name, failure)
            if failure is error:
                raise
            raise failure from error
        _report_success(run, tool_name, executed=attempt.executed)
        return result

    def _handle_tool_call(
        self, request: ToolCallRequest, handler: ToolHandler, run: RunContext
    ) -> ToolMessage | Command[Any]:
        """Enforce the policy around one model-requested tool call."""
        prepared = self._prepared(request, run)
        if isinstance(prepared, ToolMessage):
            return prepared
        request, name, arguments, key = prepared
        if key is None:
            try:
                self._gate(name, arguments)
            except PolicyViolationError as denied:
                return self._denied(request, name, denied, run)
            return handler(request)
        while True:
            attempt = _Attempt()
            operation = attempt.wrap(lambda: _tool_payload(name, handler(request)))
            try:
                payload = self._decided(name, arguments, key, operation, run)
            except _PRE_EXECUTION_DENIALS as denied:
                return self._denied(request, name, denied, run)
            except GraphBubbleUp:
                raise
            except Exception as error:
                self._await_reconciliation(name, key, error, attempt, run)
                continue
            _report_success(run, name, executed=attempt.executed)
            return _tool_message(request, name, payload)

    async def _ahandle_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
        run: RunContext,
    ) -> ToolMessage | Command[Any]:
        """Asynchronously enforce the policy around one model-requested tool call."""
        prepared = self._prepared(request, run)
        if isinstance(prepared, ToolMessage):
            return prepared
        request, name, arguments, key = prepared
        if key is None:
            try:
                self._gate(name, arguments)
            except PolicyViolationError as denied:
                return self._denied(request, name, denied, run)
            return await handler(request)

        async def call() -> dict[str, Any]:
            return _tool_payload(name, await handler(request))

        while True:
            attempt = _Attempt()
            operation = attempt.awrap(call)
            try:
                payload = await self._adecided(name, arguments, key, operation, run)
            except _PRE_EXECUTION_DENIALS as denied:
                return self._denied(request, name, denied, run)
            except GraphBubbleUp:
                raise
            except Exception as error:
                self._await_reconciliation(name, key, error, attempt, run)
                continue
            _report_success(run, name, executed=attempt.executed)
            return _tool_message(request, name, payload)

    def _prepared(
        self, request: ToolCallRequest, run: RunContext
    ) -> tuple[ToolCallRequest, str, dict[str, Any], str | None] | ToolMessage:
        """Validate a tool call so the policy judges the values the tool receives.

        Returns the request carrying the input to pass on to the tool, the tool
        name, the JSON arguments for the policy, and the idempotency key of a
        consequential tool (`None` otherwise); or an error message for the model
        when the call is invalid, in which case nothing runs and no action is
        recorded.
        """
        name = request.tool_call["name"]
        raw = request.tool_call.get("args") or {}
        tool = request.tool
        if tool is None:
            # The agent has no such tool; LangChain reports that and runs nothing.
            return request, name, dict(raw), None
        try:
            arguments = call_arguments(tool, raw)
        except InvalidArguments as invalid:
            return _rejected(request, name, f"Invalid arguments: {invalid}.", run)
        validated = request.override(
            tool_call={**request.tool_call, "args": arguments.values}
        )
        selector = self._consequential.get(name)
        if selector is None:
            return validated, name, arguments.policy, None
        if isinstance(selector, str):
            value = arguments.policy.get(selector)
            if not isinstance(value, str) or not value.strip():
                return _rejected(
                    request,
                    name,
                    f"The argument {selector!r} identifies this action and must be "
                    "a non-empty string.",
                    run,
                )
        key = _select_key(selector, arguments.policy, name)
        return validated, name, arguments.policy, key

    def _decided(
        self,
        name: str,
        arguments: Mapping[str, Any],
        key: str,
        operation: Callable[[], Any],
        run: RunContext,
    ) -> Any:
        """Run one consequential call, pausing for approval.

        A duplicate call made while the action executes elsewhere waits for that
        execution and replays its result; when the execution is abandoned,
        agentbarrier reports the outcome as unknown.
        """
        waits = 0
        while True:
            try:
                return self._barrier.execute(
                    tool_name=name,
                    arguments=arguments,
                    idempotency_key=key,
                    operation=operation,
                )
            except ApprovalRequired as pending:
                _await_approval(pending, run)
            except ActionInProgress:
                run.check_cancelled()
                time.sleep(_backoff(waits))
                waits += 1

    async def _adecided(
        self,
        name: str,
        arguments: Mapping[str, Any],
        key: str,
        operation: Callable[[], Awaitable[Any]],
        run: RunContext,
    ) -> Any:
        """Asynchronously run one consequential call, pausing for approval."""
        waits = 0
        while True:
            try:
                return await self._barrier.execute_async(
                    tool_name=name,
                    arguments=arguments,
                    idempotency_key=key,
                    operation=operation,
                )
            except ApprovalRequired as pending:
                _await_approval(pending, run)
            except ActionInProgress:
                run.check_cancelled()
                await asyncio.sleep(_backoff(waits))
                waits += 1

    def _await_reconciliation(
        self,
        tool_name: str,
        key: str,
        error: Exception,
        attempt: _Attempt,
        run: RunContext,
    ) -> None:
        """Pause a checkpointed run until an unknown outcome is reconciled.

        Returns only when a resumed run should try the call again; otherwise
        raises the failure. Only the call whose operation raised reports
        `ActionFailed`: a call refused because the action still awaits
        reconciliation is not a new failure.
        """
        failure = self._failure(tool_name, key, error, attempt)
        if not (run.pausable and isinstance(failure, ActionOutcomeUnknown)):
            _report_failure(run, tool_name, failure)
            if failure is error:
                raise error
            raise failure from error
        if failure is not error:
            _report_failure(run, tool_name, failure)
        _pause(run, reconciliation_payload(failure.action))

    def _failure(
        self, tool_name: str, key: str, error: Exception, attempt: _Attempt
    ) -> Exception:
        """Return the error to raise: the unknown action when the operation failed.

        agentbarrier marks an action whose operation raised as having an unknown
        outcome; raising `ActionOutcomeUnknown` hands that action to the caller.
        """
        if not attempt.raised(error):
            return error
        barrier = self._barrier
        for action in barrier.store.list_actions(status=RuntimeStatus.UNKNOWN):
            if (
                action.namespace == barrier.namespace
                and action.organization_id == barrier.organization_id
                and action.tool_name == tool_name
                and action.idempotency_key == key
            ):
                return ActionOutcomeUnknown(action)
        return error

    def _check(
        self, tool_name: str, arguments: Mapping[str, Any], run: RunContext | None
    ) -> None:
        try:
            self._gate(tool_name, arguments)
        except PolicyViolationError as denied:
            _report_failure(run, tool_name, denied)
            raise

    def _gate(self, tool_name: str, arguments: Mapping[str, Any]) -> None:
        decision = self._policy.evaluate(tool_name, arguments)
        if decision.effect is PolicyEffect.ALLOW:
            return
        rule = None if decision.rule_name == _DEFAULT_RULE else decision.rule_name
        if decision.effect is PolicyEffect.DENY:
            raise PolicyViolationError(tool_name, rule)
        raise PolicyConfigurationError(
            f"policy rule {decision.rule_name!r} requires approval for tool "
            f"{tool_name!r}, which is not declared consequential; declare it in "
            "PolicyManager(consequential=...) with a business idempotency key"
        )

    def _denied(
        self,
        request: ToolCallRequest,
        name: str,
        denied: Exception,
        run: RunContext,
    ) -> ToolMessage:
        _emit(run, EventName.TOOL_CALL_DENIED, tool=name, reason=type(denied).__name__)
        if self._on_denied == "raise":
            raise denied
        return ToolMessage(
            content=f"The call to {name!r} was not executed: {denied}",
            tool_call_id=request.tool_call.get("id") or "",
            name=name,
            status="error",
        )


def approval_payload(action: RuntimeAction) -> dict[str, Any]:
    """Return the JSON interrupt payload describing a pending approval."""
    return {
        "type": APPROVAL_INTERRUPT,
        "action_id": action.action_id,
        "tool": action.tool_name,
        "arguments": dict(action.arguments),
        "policy_rule": action.policy_rule,
        "namespace": action.namespace,
        "organization_id": action.organization_id,
        "expires_at_ns": action.expires_at_ns,
    }


def reconciliation_payload(action: RuntimeAction) -> dict[str, Any]:
    """Return the JSON interrupt payload describing an action with an unknown outcome.

    `error` is the type name of the exception the tool raised; the action
    waits until an operator reconciles it in agentbarrier.
    """
    return {
        "type": RECONCILIATION_INTERRUPT,
        "action_id": action.action_id,
        "tool": action.tool_name,
        "arguments": dict(action.arguments),
        "error": action.error,
        "namespace": action.namespace,
        "organization_id": action.organization_id,
    }


def _load_policy(
    policy: RuntimePolicy | str | os.PathLike[str] | Mapping[str, Any],
) -> RuntimePolicy:
    if isinstance(policy, RuntimePolicy):
        return policy
    try:
        if isinstance(policy, str | os.PathLike):
            return RuntimePolicy.from_file(Path(policy))
        if isinstance(policy, Mapping):
            return RuntimePolicy.from_mapping(policy)
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid runtime policy: {error}") from error
    raise ConfigurationError(
        "policy must be a RuntimePolicy, a policy file path, or a policy mapping"
    )


def _select_key(
    selector: IdempotencySelector, arguments: Mapping[str, Any], tool_name: str
) -> str:
    if isinstance(selector, str):
        value = arguments.get(selector)
        if not isinstance(value, str):
            raise PolicyConfigurationError(
                f"idempotency argument {selector!r} of {tool_name!r} must be a string"
            )
        key = value
    else:
        key = selector(arguments)
        if not isinstance(key, str):
            raise PolicyConfigurationError(
                f"idempotency selector of {tool_name!r} must return a string"
            )
    if not key.strip():
        raise PolicyConfigurationError(f"idempotency key of {tool_name!r} is blank")
    return key


def _rejected(
    request: ToolCallRequest, name: str, reason: str, run: RunContext
) -> ToolMessage:
    """Return an invalid call to the model; the tool does not run."""
    _emit(run, EventName.TOOL_CALL_DENIED, tool=name, reason="InvalidArguments")
    return ToolMessage(
        content=f"The call to {name!r} was not executed. {reason} "
        "Fix the arguments and call the tool again.",
        tool_call_id=request.tool_call.get("id") or "",
        name=name,
        status="error",
    )


def _backoff(waits: int) -> float:
    """Seconds to wait before checking a busy action again."""
    return min(_MAX_WAIT_SECONDS, _FIRST_WAIT_SECONDS * 2**waits)


def _tool_payload(name: str, result: ToolMessage | Command[Any]) -> dict[str, Any]:
    if not isinstance(result, ToolMessage):
        raise PolicyConfigurationError(
            f"consequential tool {name!r} returned a graph Command; consequential "
            "tools must return plain results"
        )
    return {"content": result.content, "status": result.status}


def _tool_message(request: ToolCallRequest, name: str, payload: Any) -> ToolMessage:
    # Agloom stores {"content", "status"}; results that operators reconcile in
    # agentbarrier may have any JSON shape, which the model sees as JSON text.
    if isinstance(payload, Mapping) and "content" in payload:
        content, status = payload["content"], payload.get("status", "success")
    elif isinstance(payload, str):
        content, status = payload, "success"
    else:
        content, status = json.dumps(payload, ensure_ascii=False), "success"
    return ToolMessage(
        content=content,
        status=status,
        tool_call_id=request.tool_call.get("id") or "",
        name=name,
    )


def _await_approval(pending: ApprovalRequired, run: RunContext) -> None:
    if not run.pausable:
        raise pending
    _pause(
        run,
        approval_payload(pending.action),
        EventName.APPROVAL_REQUIRED,
        tool=pending.action.tool_name,
        action_id=pending.action.action_id,
    )


def _pause(
    run: RunContext,
    payload: dict[str, Any],
    event: str | None = None,
    **details: Any,
) -> None:
    """Interrupt the run, or return when a resumed run passes this point again.

    The event is emitted only when the run actually pauses, so resumed runs do
    not report the same pause twice.
    """
    try:
        interrupt(payload)
    except GraphInterrupt:
        if event is not None:
            _emit(run, event, **details)
        raise


def _report_success(run: RunContext | None, tool_name: str, *, executed: bool) -> None:
    _emit(
        run,
        EventName.ACTION_EXECUTED if executed else EventName.ACTION_REPLAYED,
        tool=tool_name,
    )


def _report_failure(run: RunContext | None, tool_name: str, error: Exception) -> None:
    if isinstance(error, ApprovalRequired):
        _emit(
            run,
            EventName.APPROVAL_REQUIRED,
            tool=tool_name,
            action_id=error.action.action_id,
        )
    elif isinstance(error, (PolicyViolationError, *_PRE_EXECUTION_DENIALS)):
        _emit(
            run, EventName.TOOL_CALL_DENIED, tool=tool_name, reason=type(error).__name__
        )
    else:
        action = getattr(error, "action", None)
        details = {"tool": tool_name, "error": type(error).__name__}
        if isinstance(action, RuntimeAction):
            details["action_id"] = action.action_id
        _emit(run, EventName.ACTION_FAILED, **details)


def _emit(run: RunContext | None, name: str, **payload: Any) -> None:
    if run is not None:
        run.emit(name, payload)
