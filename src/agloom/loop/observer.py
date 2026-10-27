"""Deterministic normalization of iteration results into observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from agloom.loop.models import LoopObservation
from agloom.models import TopologyName

_SIGNAL_KEYS = frozenset(
    {
        "objective_progress",
        "objective_complete",
        "quality",
        "validation_passed",
        "retryable",
        "refresh_required",
        "replan_required",
        "human_intervention_required",
        "requested_topology",
        "tool_calls",
        "feedback",
    }
)
_FINGERPRINT_CHARS = 20_000
_ERROR_CHARS = 500


class DefaultLoopObserver:
    """Reads explicit loop signals from a result without making decisions.

    A result carries signals in a `loop` mapping, either as a key of a
    dictionary result or in an `AIMessage`'s `additional_kwargs`, or as
    top-level keys of a dictionary result.
    """

    def observe(
        self,
        *,
        iteration: int,
        result: Any = None,
        error: BaseException | None = None,
        topology: TopologyName | None = None,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LoopObservation:
        """Return the observation of one iteration."""
        signals = _signals(result) if error is None else {}
        details = dict(metadata or {})
        requested = signals.get("requested_topology")
        return LoopObservation(
            iteration=iteration,
            success=error is None,
            result=result,
            result_fingerprint=fingerprint(result, error),
            error_type=error_type(error),
            error_message=str(error)[:_ERROR_CHARS] if error is not None else None,
            topology=topology,
            duration=max(duration, 0.0),
            tool_calls=_tool_calls(result, signals, details),
            objective_progress=_number(signals, "objective_progress"),
            objective_complete=_flag(signals, "objective_complete"),
            quality=_number(signals, "quality"),
            validation_passed=_flag(signals, "validation_passed"),
            retryable=bool(signals.get("retryable", details.get("retryable", False))),
            refresh_required=bool(signals.get("refresh_required", False)),
            replan_required=bool(signals.get("replan_required", False)),
            human_intervention_required=bool(
                signals.get("human_intervention_required", False)
            ),
            requested_topology=(
                TopologyName(str(requested).strip().lower())
                if requested is not None
                else None
            ),
            feedback=_strings(signals.get("feedback")),
            behavior_signals=_strings(details.pop("behavior_signals", None)),
            metadata=details,
        )


def error_type(error: BaseException | None) -> str | None:
    """Return the type name of an iteration failure."""
    if error is None:
        return None
    return str(getattr(error, "error_type", None) or type(error).__name__)


def fingerprint(result: Any, error: BaseException | None) -> str:
    """Return a stable digest of a result or failure."""
    if error is not None:
        payload = f"{error_type(error)}:{str(error)[:_ERROR_CHARS]}"
    elif isinstance(result, BaseMessage):
        payload = result.text
    else:
        try:
            payload = json.dumps(result, sort_keys=True, default=_text)
        except (TypeError, ValueError):
            payload = repr(result)
    return hashlib.sha256(payload[:_FINGERPRINT_CHARS].encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return value.text if isinstance(value, BaseMessage) else repr(value)


def _signals(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        nested = result.get("loop")
        if isinstance(nested, Mapping):
            return nested
        return {key: result[key] for key in _SIGNAL_KEYS if key in result}
    if isinstance(result, AIMessage):
        nested = result.additional_kwargs.get("loop")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _tool_calls(
    result: Any, signals: Mapping[str, Any], metadata: Mapping[str, Any]
) -> int:
    explicit = signals.get("tool_calls", metadata.get("tool_calls"))
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return max(explicit, 0)
    return len(result.tool_calls) if isinstance(result, AIMessage) else 0


def _number(signals: Mapping[str, Any], name: str) -> float | None:
    value = signals.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"loop signal {name!r} must be a number")
    return float(value)


def _flag(signals: Mapping[str, Any], name: str) -> bool | None:
    value = signals.get(name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"loop signal {name!r} must be a boolean")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ValueError("loop feedback and behavior signals must be strings or sequences")
