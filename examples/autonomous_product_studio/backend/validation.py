from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .models import AcceptanceCriterion, Evidence
from .safety import INTERNAL_DIRECTORY, SafeWorkspace, WorkspaceViolation


@dataclass(frozen=True)
class ValidationResult:
    criterion_id: str
    source: str
    passed: bool
    summary: str
    details: dict[str, Any]


class ObjectiveValidator:
    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace

    def validate(self, criteria: list[AcceptanceCriterion]) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for criterion in criteria:
            if criterion.kind == "file":
                exists = self.workspace.resolve(criterion.target).is_file()
                results.append(
                    ValidationResult(
                        criterion.id,
                        "filesystem",
                        exists,
                        f"{criterion.target} {'exists' if exists else 'is missing'}",
                        {},
                    )
                )
            elif criterion.kind == "content":
                try:
                    content = self.workspace.read(criterion.target)
                    passed = bool(criterion.expected and criterion.expected in content)
                except FileNotFoundError:
                    passed, content = False, ""
                results.append(
                    ValidationResult(
                        criterion.id,
                        "filesystem",
                        passed,
                        f"expected content {'found' if passed else 'not found'} in {criterion.target}",
                        {
                            "expected": criterion.expected,
                            "bytes_inspected": len(content),
                        },
                    )
                )
            elif criterion.kind == "command":
                try:
                    outcome = self.workspace.run_check(criterion.target)
                except WorkspaceViolation as error:
                    outcome = {"passed": False, "error": str(error)}
                results.append(
                    ValidationResult(
                        criterion.id,
                        "command",
                        bool(outcome["passed"]),
                        f"allowlisted check {criterion.target} {'passed' if outcome['passed'] else 'failed'}",
                        outcome,
                    )
                )
            else:
                marker = self.workspace.resolve(f"{INTERNAL_DIRECTORY}/playwright.json")
                payload: dict[str, Any] = {}
                if marker.is_file():
                    try:
                        value = json.loads(marker.read_text(encoding="utf-8"))
                        payload = value if isinstance(value, dict) else {}
                    except json.JSONDecodeError:
                        payload = {}
                snapshot = payload.get("snapshot")
                observed_url = str(payload.get("url", "")).rstrip("/")
                expected_url = criterion.target.rstrip("/")
                passed = bool(
                    payload.get("navigation_succeeded") is True
                    and observed_url == expected_url
                    and criterion.expected
                    and isinstance(snapshot, str)
                    and criterion.expected in snapshot
                )
                results.append(
                    ValidationResult(
                        criterion.id,
                        "playwright",
                        passed,
                        f"Playwright evidence for {criterion.target} {'passed' if passed else 'is pending'}",
                        {
                            "marker": str(
                                marker.relative_to(self.workspace.root)
                            ).replace("\\", "/"),
                            "expected_url": criterion.target,
                            "observed_url": observed_url,
                            "expected_content": criterion.expected,
                            "content_matched": bool(
                                criterion.expected
                                and isinstance(snapshot, str)
                                and criterion.expected in snapshot
                            ),
                        },
                    )
                )
        return results


def persist_validation(
    store: Any, run_id: UUID, iteration: int, results: list[ValidationResult]
) -> None:
    for result in results:
        store.add_evidence(
            run_id,
            iteration,
            result.criterion_id,
            result.source,
            result.passed,
            result.summary,
            result.details,
        )


def release_is_current(workspace: SafeWorkspace) -> bool:
    """Whether this run published exactly the files currently in its workspace."""
    marker = workspace.read_release_marker()
    return marker is not None and marker.get("fingerprint") == workspace.fingerprint()


def validation_feedback(evidence: list[Evidence], *, released: bool) -> dict[str, Any]:
    """Deterministic feedback from the latest persisted validation of a run."""
    if not evidence:
        return {
            "iteration": 0,
            "instruction": "No validation has run yet.",
            "failed_criteria": [],
            "release_published": released,
            "results": [],
        }
    iteration = max(item.iteration for item in evidence)
    latest = [item for item in evidence if item.iteration == iteration]
    failed = [item.criterion_id for item in latest if not item.passed]
    if failed:
        instruction = "Repair every failed criterion, then rerun its validation."
    elif released:
        instruction = "All criteria passed and the release is published."
    else:
        instruction = (
            "All criteria passed. Call publish_release to request release approval."
        )
    return {
        "iteration": iteration,
        "instruction": instruction,
        "failed_criteria": failed,
        "release_published": released,
        "results": [
            {
                "criterion_id": item.criterion_id,
                "source": item.source,
                "passed": item.passed,
                "summary": item.summary,
                "details": item.details,
            }
            for item in latest
        ],
    }
