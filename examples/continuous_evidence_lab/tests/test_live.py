"""The main flow against the real model, MCP servers, embeddings, and PostgreSQL.

Opt in with `EVIDENCE_LIVE_TESTS=1` and set `DATABASE_URL` and `OLLAMA_URL`. The
model settings (`EXPLABS_*`) come from the environment or the repository's
`.env` file and are never printed. The flows share one event loop, like a
worker process: langchain-openai keeps one async HTTP client per process, and
its pooled connections belong to the loop that opened them.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from backend.agent import chat_model
from backend.app import create_app
from backend.auth import Principal, StaticAuthenticator, principal_scope
from backend.config import Settings
from backend.jobs import RUNNER_CONTEXT
from backend.runtime import (
    ApiResources,
    WorkerRuntime,
    open_api_resources,
    open_worker_runtime,
)
from backend.setup_database import setup_database
from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel

pytestmark = [
    pytest.mark.live,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        os.getenv("EVIDENCE_LIVE_TESTS") != "1", reason="set EVIDENCE_LIVE_TESTS=1"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEWER = Principal("live-reviewer", "live-tenant")
HEADERS = {"Authorization": "Bearer live-token"}
RUN = {
    "objective": "Verify Acme's current security, incident response, and "
    "compliance posture.",
    "topics": ["encryption", "incident response", "SOC 2"],
}


class CallRecorder(AsyncCallbackHandler):
    """Counts model calls and announces one of them."""

    def __init__(self, announce_at: int = 0) -> None:
        self.sizes: list[int] = []
        self.initial_calls = 0
        self.announce_at = announce_at
        self.announced = asyncio.Event()

    async def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        conversation = messages[0]
        self.sizes.append(len(conversation))
        # The first call of a run sees only the system prompt and the objective.
        if len(conversation) == 2 and "Loop Engineering context" not in str(
            conversation[-1].content
        ):
            self.initial_calls += 1
        if len(self.sizes) == self.announce_at:
            self.announced.set()


def recorded_model(recorder: CallRecorder) -> BaseChatModel:
    model = chat_model()
    model.callbacks = [recorder]
    return model


@pytest.fixture
def settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("EXPLABS_API_KEY"):
        pytest.skip("the model credentials are not configured")
    return Settings(database_url=os.environ["DATABASE_URL"], max_attempts=2)


@asynccontextmanager
async def live_api(
    settings: Settings,
) -> AsyncIterator[tuple[httpx.AsyncClient, ApiResources]]:
    await setup_database(settings)
    async with open_api_resources(settings) as resources:
        app = create_app(
            settings=settings,
            resources=resources,
            authenticator=StaticAuthenticator(REVIEWER, token="live-token"),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://evidence.live",
                headers=HEADERS,
                timeout=60,
            ) as client,
        ):
            yield client, resources


def worker_for(settings: Settings, resources: ApiResources, recorder: CallRecorder):
    return open_worker_runtime(
        settings,
        store=resources.store,
        jobs=resources.jobs,
        barrier=resources.approvals.store,
        model_factory=lambda: recorded_model(recorder),
    )


async def drain(worker: WorkerRuntime) -> None:
    await worker.jobs.run_worker_async(
        queues=[worker.settings.queue],
        wait=False,
        additional_context={RUNNER_CONTEXT: worker.runner},
        install_signal_handlers=False,
    )


def summary(view: dict[str, Any]) -> str:
    return (
        f"status={view['status']} execution={view['execution_status']} "
        f"termination={view['termination']} quality={view['accuracy']['quality']} "
        f"coverage={view['accuracy']['coverage']} "
        f"precision={view['accuracy']['citation_precision']} "
        f"timeline={[item['action'] for item in view['timeline']]}"
    )


async def approve_and_publish(
    client: httpx.AsyncClient, worker: WorkerRuntime, run_id: str
) -> dict[str, Any]:
    decided = await client.post(
        f"/api/runs/{run_id}/decision", json={"decision": "approve", "note": "live"}
    )
    assert decided.status_code == 200, decided.text
    await drain(worker)
    return (await client.get(f"/api/runs/{run_id}")).json()


async def test_live_review_flow_publishes_after_approval(settings: Settings):
    recorder = CallRecorder()
    async with (
        live_api(settings) as (client, resources),
        worker_for(settings, resources, recorder) as worker,
    ):
        run_id = (await client.post("/api/runs", json=RUN)).json()["run_id"]
        await drain(worker)
        paused = (await client.get(f"/api/runs/{run_id}")).json()
        print(
            f"\nreview flow paused: {summary(paused)} model_calls={len(recorder.sizes)}"
        )
        assert paused["status"] == "awaiting_approval", summary(paused)
        assert paused["approval"]["policy_rule"] == (
            "reviewer approves every evidence publication"
        )

        done = await approve_and_publish(client, worker, run_id)
        print(
            f"review flow finished: {summary(done)} model_calls={len(recorder.sizes)}"
        )
        assert done["status"] == "approved", summary(done)
        assert done["published_at"] is not None
        with principal_scope(REVIEWER):
            events = await resources.store.read_events(UUID(run_id))
        print(
            "events:", [event["type"] for event in events if event["type"] != "runtime"]
        )


async def test_live_crash_resumes_from_the_checkpoint(settings: Settings):
    recorder = CallRecorder(announce_at=2)
    async with live_api(settings) as (client, resources):
        run_id = (await client.post("/api/runs", json=RUN)).json()["run_id"]
        async with worker_for(settings, resources, recorder) as first:
            with principal_scope(REVIEWER):
                crashed = asyncio.create_task(first.runner.execute(UUID(run_id)))
            await asyncio.wait_for(recorder.announced.wait(), 600)
            crashed.cancel()
            with pytest.raises(asyncio.CancelledError):
                await crashed
        before = len(recorder.sizes)
        print(f"\ncrashed during model call {before}")

        async with worker_for(settings, resources, recorder) as second:
            await drain(second)
            resumed = (await client.get(f"/api/runs/{run_id}")).json()
            print(f"resumed: {summary(resumed)} model_calls={len(recorder.sizes)}")
            with principal_scope(REVIEWER):
                event_types = [
                    event["type"]
                    for event in await resources.store.read_events(UUID(run_id))
                ]
            assert "resumed" in event_types
            # The run's first model call is not repeated after the crash.
            assert recorder.initial_calls == 1, recorder.sizes
            if resumed["status"] == "awaiting_approval":
                done = await approve_and_publish(client, second, run_id)
                print(f"crash flow finished: {summary(done)}")
                assert done["status"] == "approved", summary(done)
            else:
                assert resumed["status"] in {"unvalidated", "approved"}, summary(
                    resumed
                )
