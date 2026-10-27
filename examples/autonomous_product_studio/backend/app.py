"""Product Studio HTTP API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from agentbarrier import Decision
from agentbarrier.errors import (
    ApprovalAuthorizationError,
    ApprovalExpired,
    InvalidActionState,
    RuntimeStoreError,
)
from agentbarrier.runtime import RuntimeReconciliation
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import (
    Authenticator,
    OIDCAuthenticator,
    current_principal,
    principal_scope,
)
from .governance import (
    POLICY_NAMESPACE,
    reconciliation_refusal,
    reviewer,
    with_decision,
)
from .mcp import PLAYWRIGHT_SERVER_ID
from .models import (
    ACTIVE_RUN_STATUSES,
    Approval,
    ApprovalDecision,
    LoopRun,
    Project,
    ProjectCreate,
    ReconciliationDecision,
    Task,
    TaskCreate,
)
from .observability import observability_status
from .services import StudioServices, open_services
from .settings import APP_ROOT, StudioSettings
from .worker import run_worker

SEED_SPEC = APP_ROOT / "seed" / "product_spec.json"
_LOGGER = logging.getLogger(__name__)
ServicesFactory = Callable[[], AbstractAsyncContextManager[StudioServices]]
_DECIDED = {"approved", "succeeded"}


def production_services() -> AbstractAsyncContextManager[StudioServices]:
    settings = StudioSettings.from_env()
    return open_services(settings, run_agents=settings.embedded_worker)


def _seed(services: StudioServices) -> None:
    store = services.store
    if store.list_projects() or not SEED_SPEC.is_file():
        return
    store.create_project(
        ProjectCreate.model_validate_json(Path(SEED_SPEC).read_text(encoding="utf-8"))
    )


def create_app(
    services: ServicesFactory | None = None,
    authenticator: Authenticator | None = None,
) -> FastAPI:
    open_studio = services or production_services
    token_authenticator = authenticator or OIDCAuthenticator()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        token_authenticator.verify_configuration()
        async with open_studio() as opened:
            app.state.services = opened
            worker: asyncio.Task[None] | None = None
            if opened.settings.embedded_worker:
                worker = asyncio.create_task(
                    run_worker(opened, install_signal_handlers=False)
                )
            try:
                yield
            finally:
                if worker is not None:
                    worker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await worker

    app = FastAPI(title="Autonomous Product Studio", version="1.0.1", lifespan=lifespan)

    def studio(request: Request) -> StudioServices:
        return request.app.state.services

    @app.middleware("http")
    async def require_authentication(request: Request, call_next: Any) -> Any:
        if request.url.path in {"/api/health", "/api/ready"}:
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
        allow_origins=[
            origin.strip()
            for origin in os.getenv(
                "STUDIO_CORS_ORIGINS", "http://localhost:5178"
            ).split(",")
            if origin.strip()
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        services_ = studio(request)
        return {
            "status": "ok",
            "runs_agents": services_.settings.embedded_worker,
            "observability": observability_status(),
            "mcp_servers": [PLAYWRIGHT_SERVER_ID]
            if services_.settings.browser_validation
            else [],
            "policy": {
                "namespace": POLICY_NAMESPACE,
                "version": services_.policy.version,
            },
        }

    @app.get("/api/ready")
    async def ready(request: Request) -> dict[str, str]:
        services_ = studio(request)
        try:
            await asyncio.to_thread(services_.store.ping)
            if not await services_.jobs.ping():
                raise ConnectionError("job queue unavailable")
        except Exception as exc:
            _LOGGER.warning("Studio readiness failed: %s", type(exc).__name__)
            raise HTTPException(503, "database unavailable") from exc
        return {"status": "ready"}

    @app.get("/api/projects", response_model=list[Project])
    def projects(request: Request) -> list[Project]:
        services_ = studio(request)
        _seed(services_)
        return services_.store.list_projects()

    @app.post("/api/projects", response_model=Project)
    def create_project(request: Request, payload: ProjectCreate) -> Project:
        return studio(request).store.create_project(payload)

    @app.get("/api/projects/{project_id}", response_model=Project)
    def project(request: Request, project_id: UUID) -> Project:
        value = studio(request).store.get_project(project_id)
        if value is None:
            raise HTTPException(404, "project not found")
        return value

    @app.post("/api/projects/{project_id}/tasks", response_model=Task)
    def create_task(request: Request, project_id: UUID, payload: TaskCreate) -> Task:
        try:
            return studio(request).store.create_task(project_id, payload)
        except LookupError as exc:
            raise HTTPException(404, "project not found") from exc

    @app.post("/api/tasks/{task_id}/run", response_model=LoopRun, status_code=202)
    async def run_task(
        request: Request,
        task_id: UUID,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> LoopRun:
        services_ = studio(request)
        try:
            run, created = await services_.store.schedule_run(
                task_id,
                idempotency_key=idempotency_key,
                enqueue=services_.jobs.enqueue_run,
            )
        except LookupError as exc:
            raise HTTPException(404, "task not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if created:
            await asyncio.to_thread(
                services_.events.publish,
                run.id,
                "queued",
                message="Bounded build accepted and queued.",
                run=run.model_dump(mode="json"),
            )
        return run

    @app.get("/api/runs/{run_id}", response_model=LoopRun)
    def get_run(request: Request, run_id: UUID) -> LoopRun:
        value = studio(request).store.get_run(run_id)
        if value is None:
            raise HTTPException(404, "run not found")
        return value

    @app.get("/api/runs/{run_id}/events")
    def stream_run(
        request: Request,
        run_id: UUID,
        last_event_id: int = Header(default=0, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        services_ = studio(request)
        if services_.store.get_run(run_id) is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(
            services_.events.stream(
                run_id, principal=current_principal(), after=max(last_event_id, 0)
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(request: Request, run_id: UUID) -> dict[str, str]:
        services_ = studio(request)
        store = services_.store
        run = await asyncio.to_thread(store.get_run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(409, "run is not cancellable")
        if run.status == "awaiting_reconciliation":
            raise HTTPException(
                409,
                "a release outcome is unknown; reconcile it before cancelling the run",
            )
        if run.status == "awaiting_approval":
            await asyncio.to_thread(
                _reject_pending, services_, run, "the run was cancelled"
            )
            return {"status": await _end_run(services_, run, "cancelled", {})}
        job_id = await asyncio.to_thread(store.run_job_id, run_id)
        outcome = await services_.jobs.cancel(job_id) if job_id else "finished"
        if outcome == "cancelled":
            return {"status": await _end_run(services_, run, "cancelled", {})}
        if outcome == "aborting":
            await asyncio.to_thread(
                services_.events.publish,
                run_id,
                "cancelling",
                message="Cancellation requested; the agent stops at its next step.",
            )
            return {"status": "cancelling"}
        raise HTTPException(409, "run is not cancellable")

    @app.get("/api/runs/{run_id}/evidence")
    def evidence(request: Request, run_id: UUID) -> list[Any]:
        store = studio(request).store
        if store.get_run(run_id) is None:
            raise HTTPException(404, "run not found")
        return store.evidence_for_run(run_id)

    @app.get("/api/runs/{run_id}/approvals", response_model=list[Approval])
    def approvals(request: Request, run_id: UUID) -> list[Approval]:
        services_ = studio(request)
        run = services_.store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return [_current(services_, approval) for approval in run.approvals]

    @app.post("/api/runs/{run_id}/approvals/{action_id}", response_model=LoopRun)
    async def decide(
        request: Request, run_id: UUID, action_id: str, payload: ApprovalDecision
    ) -> LoopRun:
        services_ = studio(request)
        store = services_.store
        principal = current_principal()
        run = await asyncio.to_thread(store.get_run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if not any(
            item.action_id == action_id and item.kind == "approval"
            for item in run.approvals
        ):
            raise HTTPException(404, "approval not found")
        if run.status != "awaiting_approval":
            raise HTTPException(409, "the run is not awaiting approval")
        decision = (
            Decision.APPROVE if payload.decision == "approve" else Decision.REJECT
        )
        try:
            await asyncio.to_thread(
                services_.policy_store.decide_authorized,
                action_id,
                decision,
                authorization=reviewer(
                    principal,
                    separate_approver=services_.settings.require_separate_approver,
                ),
                reason=payload.reason,
            )
        except ApprovalAuthorizationError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ApprovalExpired, InvalidActionState) as exc:
            raise HTTPException(409, str(exc)) from exc
        await asyncio.to_thread(
            services_.events.publish,
            run_id,
            "approval_decided",
            action_id=action_id,
            decision=payload.decision,
            decided_by=principal.subject,
            message=f"Release {payload.decision}d by {principal.subject}.",
        )
        if decision is Decision.REJECT:
            await _end_run(
                services_,
                run,
                "rejected",
                {"rejected_by": principal.subject, "reason": payload.reason},
            )
        else:
            current = [_current(services_, item) for item in run.approvals]
            if all(item.status in _DECIDED for item in current):
                with contextlib.suppress(ValueError):
                    await store.requeue_run(run_id, enqueue=services_.jobs.enqueue_run)
                    await asyncio.to_thread(
                        services_.events.publish,
                        run_id,
                        "queued",
                        message="Release approved; the run resumes from its checkpoint.",
                    )
        updated = await asyncio.to_thread(store.get_run, run_id)
        assert updated is not None
        return updated

    @app.post("/api/runs/{run_id}/reconciliations/{action_id}", response_model=LoopRun)
    async def reconcile(
        request: Request,
        run_id: UUID,
        action_id: str,
        payload: ReconciliationDecision,
    ) -> LoopRun:
        """Record whether a release whose tool raised took effect, then resume."""
        services_ = studio(request)
        store = services_.store
        principal = current_principal()
        run = await asyncio.to_thread(store.get_run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if not any(
            item.action_id == action_id and item.kind == "reconciliation"
            for item in run.approvals
        ):
            raise HTTPException(404, "reconciliation not found")
        if run.status != "awaiting_reconciliation":
            raise HTTPException(409, "the run is not awaiting reconciliation")
        policy_store = services_.policy_store
        try:
            action = await asyncio.to_thread(policy_store.get_action, action_id)
        except RuntimeStoreError as exc:
            raise HTTPException(404, "reconciliation not found") from exc
        refusal = reconciliation_refusal(
            action,
            principal,
            separate_approver=services_.settings.require_separate_approver,
        )
        if refusal is not None:
            raise HTTPException(403, refusal)
        committed = payload.outcome == "committed"
        result = (
            payload.result
            if payload.result is not None
            else {"reconciled": "committed", "reason": payload.reason}
        )
        try:
            await asyncio.to_thread(
                policy_store.reconcile,
                action_id,
                RuntimeReconciliation(payload.outcome),
                resolved_by=principal.subject,
                reason=payload.reason,
                result=result if committed else None,
            )
        except InvalidActionState as exc:
            raise HTTPException(409, str(exc)) from exc
        await asyncio.to_thread(
            services_.events.publish,
            run_id,
            "reconciled",
            action_id=action_id,
            outcome=payload.outcome,
            reconciled_by=principal.subject,
            message=(
                f"{principal.subject} recorded that the release "
                f"{'took effect' if committed else 'did not take effect'}."
            ),
        )
        current = [_current(services_, item) for item in run.approvals]
        if all(item.status != "unknown" for item in current):
            with contextlib.suppress(ValueError):
                await store.requeue_run(run_id, enqueue=services_.jobs.enqueue_run)
                await asyncio.to_thread(
                    services_.events.publish,
                    run_id,
                    "queued",
                    message=(
                        "Release outcome reconciled; the run resumes from its "
                        "checkpoint."
                    ),
                )
        updated = await asyncio.to_thread(store.get_run, run_id)
        assert updated is not None
        return updated

    @app.get("/api/projects/{project_id}/artifacts/search")
    def search(
        request: Request, project_id: UUID, q: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        store = studio(request).store
        if store.get_project(project_id) is None:
            raise HTTPException(404, "project not found")
        return store.search_artifacts(project_id, q, min(max(limit, 1), 20))

    return app


def _current(services: StudioServices, approval: Approval) -> Approval:
    try:
        return with_decision(
            approval, services.policy_store.get_action(approval.action_id)
        )
    except RuntimeStoreError:
        return approval


def _reject_pending(services: StudioServices, run: LoopRun, reason: str) -> None:
    principal = current_principal()
    for approval in run.approvals:
        if _current(services, approval).status != "pending":
            continue
        with contextlib.suppress(ApprovalExpired, InvalidActionState):
            services.policy_store.decide(
                approval.action_id,
                Decision.REJECT,
                decided_by=principal.subject,
                reason=reason,
            )


async def _end_run(
    services: StudioServices, run: LoopRun, status: str, result: dict[str, Any]
) -> str:
    """End a run that no job is executing, then let a worker discard its state."""
    principal = current_principal()
    proof = run.proof.model_copy(update={"termination_reason": status})
    finished = await asyncio.to_thread(
        services.store.finish_run, run.id, status, proof, result
    )
    await asyncio.to_thread(
        services.events.publish,
        run.id,
        status,
        message="Run cancelled." if status == "cancelled" else "Release rejected.",
        run=finished.model_dump(mode="json") if finished else None,
    )
    await services.jobs.enqueue_discard(run.id, principal)
    return status


app = create_app()
