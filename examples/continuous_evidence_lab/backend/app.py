"""FastAPI API: admission, run views, live events, cancellation, and reviewer decisions.

Runs execute in procrastinate workers (`python -m backend.worker`). With
`WORKER_IN_API=true` the lifespan also runs a worker in this process, with the
MCP runtime registered at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import socket
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from agentbarrier.errors import (
    ApprovalAuthorizationError,
    ApprovalExpired,
    InvalidActionState,
)
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg import AsyncConnection

from .agent import loop_policy
from .auth import (
    Authenticator,
    OIDCAuthenticator,
    current_principal,
    principal_scope,
)
from .config import Settings
from .jobs import enqueue_or_reuse
from .mcp import mcp_servers
from .models import DecisionRequest, RunRequest, RunView
from .observability import DECISIONS, RUNS, observability_status, process_telemetry
from .policy import PUBLISH_TOOL, load_policy
from .runtime import ApiResources, open_api_resources, open_worker_runtime
from .streaming import RunEventBroker

_LOGGER = logging.getLogger(__name__)
_PUBLIC_PATHS = frozenset({"/api/health", "/api/ready"})


def ensure_safe_target(url: str) -> None:
    """Reject credentials, non-HTTP schemes, and non-public destinations."""
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("target must be an unauthenticated HTTP(S) URL")
    if parsed.hostname == "target-site":
        if parsed.scheme != "http" or parsed.port not in {None, 80}:
            raise ValueError("target-site is approved only on HTTP port 80")
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as exc:
        raise ValueError("target host cannot be resolved") from exc
    if not addresses or any(
        not ipaddress.ip_address(address).is_global for address in addresses
    ):
        raise ValueError("non-public targets are not allowed")


def create_app(
    *,
    settings: Settings | None = None,
    resources: ApiResources | None = None,
    authenticator: Authenticator | None = None,
) -> FastAPI:
    """Create the API; `resources` replaces the PostgreSQL resources in tests."""
    config = settings or Settings.from_env()
    token_authenticator = authenticator or OIDCAuthenticator()
    policy = load_policy(config.policy_file)
    holder: dict[str, Any] = {}

    def platform() -> ApiResources:
        return holder["resources"]

    def broker() -> RunEventBroker:
        return holder["broker"]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        token_authenticator.verify_configuration()
        async with AsyncExitStack() as stack:
            # Serves this process's metrics; an in-process worker shares it.
            telemetry = process_telemetry("api")
            stack.callback(telemetry.shutdown)
            opened = resources or await stack.enter_async_context(
                open_api_resources(config)
            )
            holder["resources"] = opened
            holder["broker"] = RunEventBroker(opened.store)
            if config.worker_in_api:
                worker = await stack.enter_async_context(
                    open_worker_runtime(
                        config,
                        store=opened.store,
                        jobs=opened.jobs,
                        barrier=opened.approvals.store,
                    )
                )
                task = asyncio.create_task(worker.run(install_signal_handlers=False))
                stack.push_async_callback(_stop_task, task)
            yield

    app = FastAPI(title="Continuous Evidence Lab", lifespan=lifespan)

    @app.middleware("http")
    async def require_authentication(request: Request, call_next: Any) -> Any:
        if request.url.path in _PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            return JSONResponse(
                {"detail": "bearer token required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            principal = token_authenticator.authenticate(authorization[7:].strip())
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )
        request.state.principal = principal
        with principal_scope(principal):
            return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
        ],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "observability": observability_status()}

    @app.get("/api/ready")
    async def ready() -> dict[str, str]:
        try:
            await platform().store.ping()
        except Exception as exc:
            _LOGGER.warning("database readiness failed: %s", type(exc).__name__)
            raise HTTPException(503, "database unavailable") from exc
        return {"status": "ready"}

    @app.get("/api/governance")
    async def governance() -> dict[str, Any]:
        return {
            "mcp_servers": sorted(mcp_servers()),
            "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"),
            "loop": loop_policy().model_dump(),
            "approval_required": True,
            "consequential_tools": [PUBLISH_TOOL],
            "policy": {
                "version": policy.version,
                "default": policy.default_effect.value,
                "rules": [
                    {"name": rule.name, "effect": rule.effect.value, "tool": rule.tool}
                    for rule in policy.rules
                ],
            },
        }

    async def enqueue(connection: AsyncConnection[Any] | None, run_id: UUID) -> int:
        principal = current_principal()
        return await enqueue_or_reuse(
            platform().jobs,
            run_id,
            tenant_id=principal.tenant_id,
            user_id=principal.subject,
            connection=connection,
        )

    @app.post("/api/runs", status_code=202)
    async def start(
        request: RunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, str]:
        if request.target_url is not None:
            try:
                await asyncio.to_thread(ensure_safe_target, str(request.target_url))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        try:
            run_id, created = await platform().store.admit_run(
                request, idempotency_key=idempotency_key, enqueue=enqueue
            )
            if created:
                await broker().publish(
                    run_id, "queued", message="Evidence run accepted and queued."
                )
            run = await platform().store.get_run(run_id)
        except Exception as exc:
            _LOGGER.exception("evidence run admission failed")
            raise HTTPException(
                503, f"unable to schedule evidence run ({type(exc).__name__})"
            ) from exc
        if created:
            RUNS.labels("queued").inc()
        return {
            "run_id": str(run_id),
            "status": run.status if run else "running",
            "execution_status": run.execution_status if run else "queued",
        }

    @app.get("/api/runs/{run_id}", response_model=RunView)
    async def get_run(run_id: UUID) -> RunView:
        run = await platform().store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    @app.get("/api/runs/{run_id}/events")
    async def stream_run(
        run_id: UUID,
        last_event_id: int = Header(default=0, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        if await platform().store.run_state(run_id) is None:
            raise HTTPException(404, "run not found")
        principal = current_principal()

        async def events() -> AsyncGenerator[str, None]:
            with principal_scope(principal):
                async for chunk in broker().stream(run_id, after=max(last_event_id, 0)):
                    yield chunk

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: UUID) -> dict[str, str]:
        store = platform().store
        state = await store.run_state(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        if state.status in {"approved", "rejected"}:
            raise HTTPException(409, "run is not cancellable")
        principal = current_principal()
        reason = f"cancelled by {principal.subject}"
        pending_review = (
            state.execution_status == "paused"
            and state.approval is not None
            and state.approval.decision is None
        )
        if pending_review and state.approval is not None:
            await platform().approvals.withdraw(
                state.approval.action_id, actor=principal.subject
            )
            if await store.cancel_run(run_id, reason):
                return await _cancelled(run_id)
        manager = platform().jobs.job_manager
        # A paused run with a decided approval waits for its retried job.
        active = {"queued", "running", "paused"}
        if state.execution_status in active and state.job_id is not None:
            if await manager.cancel_job_by_id_async(state.job_id):
                if await store.cancel_run(run_id, reason):
                    return await _cancelled(run_id)
            elif await manager.cancel_job_by_id_async(state.job_id, abort=True):
                await broker().publish(
                    run_id, "cancelling", message="Cancellation requested."
                )
                return {"status": "cancelling"}
        raise HTTPException(409, "run is not cancellable")

    async def _cancelled(run_id: UUID) -> dict[str, str]:
        run = await platform().store.get_run(run_id)
        RUNS.labels("cancelled").inc()
        await broker().publish(
            run_id,
            "cancelled",
            message="The run was cancelled.",
            run=run.model_dump(mode="json") if run else None,
        )
        return {"status": "cancelled"}

    @app.post("/api/runs/{run_id}/decision")
    async def decide(run_id: UUID, request: DecisionRequest) -> dict[str, Any]:
        store = platform().store
        state = await store.run_state(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        approval = state.approval
        if (
            state.status != "awaiting_approval"
            or approval is None
            or approval.decision is not None
        ):
            raise HTTPException(409, "run is not awaiting approval")
        principal = current_principal()
        try:
            await platform().approvals.decide(
                approval.action_id,
                approve=request.decision == "approve",
                reviewer=principal.subject,
                tenant_id=principal.tenant_id,
                note=request.note,
            )
        except ApprovalAuthorizationError as exc:
            raise HTTPException(403, "you are not allowed to decide this run") from exc
        except ApprovalExpired as exc:
            if await store.requeue_paused(run_id, enqueue=enqueue):
                await broker().publish(
                    run_id, "decision", message="The approval window has closed."
                )
            raise HTTPException(409, "the approval window has closed") from exc
        except InvalidActionState as exc:
            raise HTTPException(409, "the action was already decided") from exc
        run = await store.record_decision(
            run_id, request.decision, request.note, enqueue=enqueue
        )
        if run is None:
            raise HTTPException(409, "run is not awaiting approval")
        DECISIONS.labels(request.decision).inc()
        await broker().publish(
            run_id,
            "decision",
            decision=request.decision,
            decided_by=principal.subject,
            message="Approved; publishing the evidence."
            if request.decision == "approve"
            else "Rejected; the evidence will not be published.",
            run=run.model_dump(mode="json"),
        )
        return {"decision": request.decision, "run": run.model_dump(mode="json")}

    return app


async def _stop_task(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def app_factory() -> FastAPI:
    """Uvicorn factory: `uvicorn backend.app:app_factory --factory`."""
    return create_app()
