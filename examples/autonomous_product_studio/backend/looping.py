from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from agloom import DefaultLoopObserver, LoopObservation, TopologyName

from .models import AcceptanceCriterion
from .safety import SafeWorkspace
from .validation import ObjectiveValidator, persist_validation, release_is_current

IterationListener = Callable[[dict[str, Any]], None]


class ValidationLoopObserver(DefaultLoopObserver):
    """Turns objective evidence into Loop Engineering signals after each iteration.

    An iteration completes the objective only when every acceptance criterion
    passed and the run published exactly the validated files. Evidence is
    persisted per iteration, so the latest results survive worker restarts.
    """

    def __init__(
        self,
        validator: ObjectiveValidator,
        store: Any,
        run_id: UUID,
        criteria: list[AcceptanceCriterion],
        workspace: SafeWorkspace,
        on_iteration: IterationListener | None = None,
    ) -> None:
        self.validator = validator
        self.store = store
        self.run_id = run_id
        self.criteria = criteria
        self.workspace = workspace
        self.on_iteration = on_iteration

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
        base = super().observe(
            iteration=iteration,
            result=result,
            error=error,
            topology=topology,
            duration=duration,
            metadata=metadata,
        )
        validations = self.validator.validate(self.criteria)
        persist_validation(self.store, self.run_id, iteration, validations)
        passed = sum(item.passed for item in validations)
        total = len(validations)
        rate = passed / total if total else 0.0
        released = release_is_current(self.workspace)
        complete = bool(total) and passed == total and released and error is None
        progress = (passed + int(released and passed == total)) / (total + 1)
        failed = [item.criterion_id for item in validations if not item.passed]
        if self.on_iteration is not None:
            self.on_iteration(
                {
                    "iteration": iteration,
                    "progress": progress,
                    "quality": rate,
                    "passed_criteria": passed,
                    "acceptance_criteria": total,
                    "failed_criteria": failed,
                    "release_published": released,
                    "message": _message(passed, total, released),
                }
            )
        return base.model_copy(
            update={
                "objective_progress": progress,
                "objective_complete": complete,
                "validation_passed": complete,
                "quality": rate,
                "retryable": error is not None or not complete,
                "metadata": {
                    **base.metadata,
                    "acceptance_criteria": total,
                    "passed_criteria": passed,
                    "validation_pass_rate": rate,
                    "release_published": released,
                },
            }
        )


def _message(passed: int, total: int, released: bool) -> str:
    if passed < total:
        return f"{passed} of {total} acceptance criteria passed."
    if released:
        return "All acceptance criteria passed and the release is published."
    return "All acceptance criteria passed; the release is not published yet."
