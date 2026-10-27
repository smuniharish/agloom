"""HTTP API of the Web Compliance Auditor."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .auditor import AuditError, Auditor
from .governance import RECORD_AUDIT
from .mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from .models import ApprovalRequest, Audit, AuditRequest, FeedbackRequest
from .runtime import Runtime, production_runtime

RuntimeFactory = Callable[[], AbstractAsyncContextManager[Runtime]]


def current_runtime(request: Request) -> Runtime:
    """Return the runtime started by the application lifespan."""
    return request.app.state.runtime


def current_auditor(runtime: Annotated[Runtime, Depends(current_runtime)]) -> Auditor:
    """Return the auditor, or fail when no model is configured."""
    if runtime.auditor is None:
        raise HTTPException(503, "EXPLABS_API_KEY is not configured")
    return runtime.auditor


RuntimeDependency = Annotated[Runtime, Depends(current_runtime)]
AuditorDependency = Annotated[Auditor, Depends(current_auditor)]


def create_app(runtime_factory: RuntimeFactory = production_runtime) -> FastAPI:
    """Create the API; the runtime starts and stops with the application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory() as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="Web Compliance Auditor", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(AuditError)
    async def audit_error(request: Request, error: AuditError) -> JSONResponse:
        del request
        return JSONResponse({"detail": error.detail}, status_code=error.status)

    @app.get("/api/health")
    async def health(runtime: RuntimeDependency) -> dict[str, Any]:
        counts = await asyncio.to_thread(runtime.backends.counts)
        return {
            "status": "ok" if runtime.agent is not None else "unconfigured",
            "model_configured": runtime.agent is not None,
            "database": dict(counts),
            "mcp_servers": list(runtime.mcp.servers),
            "observability": runtime.telemetry.status(),
            "policy_version": runtime.policy.version,
        }

    @app.get("/api/capabilities")
    async def capabilities(runtime: RuntimeDependency) -> dict[str, Any]:
        records = await runtime.mcp.registry.list()
        return {
            "servers": {
                server: str(await runtime.mcp.health(server))
                for server in runtime.mcp.servers
            },
            "capabilities": [
                {
                    "id": record.capability_id,
                    "server": record.server_id,
                    "type": str(record.type),
                    "name": record.name,
                }
                for record in records
            ],
        }

    @app.get("/api/governance")
    async def governance(runtime: RuntimeDependency) -> dict[str, Any]:
        policy = runtime.policy
        return {
            "grounding": "retrieved policy IDs only",
            "mcp_servers": {
                "browser": PLAYWRIGHT,
                "filesystem": FILESYSTEM,
                "reference": EVERYTHING,
            },
            "policy_version": policy.version,
            "default_effect": policy.default_effect.value,
            "consequential": {RECORD_AUDIT: "audit_id"},
            "rules": [
                {
                    "name": rule.name,
                    "effect": rule.effect.value,
                    "tool": rule.tool,
                    "conditions": [
                        {
                            "path": condition.path,
                            "operator": condition.operator.value,
                            "value": condition.value,
                        }
                        for condition in rule.conditions
                    ],
                }
                for rule in policy.rules
            ],
        }

    @app.post("/api/audits", response_model=Audit)
    async def audit(request: AuditRequest, auditor: AuditorDependency) -> Audit:
        return await auditor.audit(request)

    @app.get("/api/audits/{audit_id}", response_model=Audit)
    async def stored_audit(audit_id: UUID, auditor: AuditorDependency) -> Audit:
        return await auditor.get(audit_id)

    @app.post("/api/audits/{audit_id}/approval", response_model=Audit)
    async def approval(
        audit_id: UUID, request: ApprovalRequest, auditor: AuditorDependency
    ) -> Audit:
        return await auditor.review(audit_id, request)

    @app.post("/api/audits/{audit_id}/record", response_model=Audit)
    async def record(audit_id: UUID, auditor: AuditorDependency) -> Audit:
        return await auditor.record_reconciled(audit_id)

    @app.post("/api/audits/{audit_id}/feedback")
    async def feedback(
        audit_id: UUID, request: FeedbackRequest, auditor: AuditorDependency
    ) -> dict[str, str]:
        return await auditor.submit_feedback(audit_id, request)

    @app.post("/api/policies/refresh")
    async def refresh(runtime: RuntimeDependency) -> dict[str, Any]:
        result = await runtime.refresh.refresh()
        return {
            "status": str(result.status),
            "discovered": result.discovered_count,
            "added": result.added_count,
            "modified": result.modified_count,
            "deleted": result.deleted_count,
            "refreshed": result.refreshed_count,
        }

    return app


app = create_app()
