from uuid import uuid4

import httpx
import pytest
from backend.models import Accuracy, Approval, Iteration, RunRequest, Synthesis
from backend.store import OllamaEmbeddings

from tests.support import MemoryStore, synthesis


async def _enqueue(connection, run_id) -> int:
    return 1


async def test_run_timeline_review_and_publication_lifecycle():
    store = MemoryStore()
    run_id, created = await store.admit_run(
        RunRequest(objective="Verify controls", topics=["encryption"]),
        idempotency_key="key-1",
        enqueue=_enqueue,
    )
    assert created
    assert (
        await store.admit_run(
            RunRequest(objective="Verify controls", topics=["encryption"]),
            idempotency_key="key-1",
            enqueue=_enqueue,
        )
    ) == (run_id, False)
    await store.save_iteration(
        run_id,
        Iteration(
            iteration=1,
            action="replan",
            progress=0.5,
            quality=0.4,
            objective_complete=False,
            gaps=["encryption"],
            reason="coverage gap",
        ),
        {"action": "replan"},
    )
    result = Synthesis.model_validate(synthesis(["encryption"]))
    perfect = Accuracy(citation_precision=1, claim_support=1, coverage=1, quality=1)
    await store.await_approval(
        run_id,
        result,
        perfect,
        Approval(action_id=str(uuid4()), tool="publish_evidence", policy_rule="review"),
    )
    paused = await store.get_run(run_id)
    assert paused is not None
    assert paused.timeline[0].action == "replan"
    assert paused.status == "awaiting_approval"
    assert paused.execution_status == "paused"
    assert [source.id for source in paused.sources] == []
    await store.search("encryption")
    assert [source.id for source in (await store.get_run(run_id)).sources] == [
        "VENDOR-SEC-001"
    ]
    decided = await store.record_decision(run_id, "approve", "ok", enqueue=_enqueue)
    assert decided is not None
    assert decided.approval.decision == "approve"
    assert await store.record_decision(run_id, "reject", None, enqueue=_enqueue) is None
    first = await store.publish(run_id, result, perfect)
    assert await store.publish(run_id, result, perfect) == first
    assert store.publications == [run_id]
    await store.complete_run(
        run_id, termination="validated", synthesis=None, accuracy=None
    )
    done = await store.get_run(run_id)
    assert done.status == "approved"
    assert done.objective_state == "validated"
    assert done.execution_status == "completed"


async def test_rejected_and_cancelled_runs_keep_their_outcome():
    store = MemoryStore()
    run_id, _ = await store.admit_run(
        RunRequest(objective="Verify controls", topics=["encryption"]),
        idempotency_key=None,
        enqueue=_enqueue,
    )
    assert await store.cancel_run(run_id, "cancelled by test-user")
    await store.fail_run(run_id, "late failure")
    view = await store.get_run(run_id)
    assert view.status == "cancelled"
    assert not await store.cancel_run(run_id, "again")


async def test_unreachable_embedding_service_does_not_return_a_vector(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    with pytest.raises(httpx.ConnectError):
        await OllamaEmbeddings().embed(["security evidence"])
