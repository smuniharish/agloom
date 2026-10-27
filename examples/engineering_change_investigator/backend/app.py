"""HTTP API of the Engineering Change Investigator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .governance import RELEASE_DECISION
from .investigator import InvestigationError, Investigator
from .models import (
    ApprovalRequest,
    FeedbackRequest,
    Investigation,
    InvestigationRequest,
)
from .runtime import Runtime, production_runtime

RuntimeFactory = Callable[[], AbstractAsyncContextManager[Runtime]]


def current_runtime(request: Request) -> Runtime:
    """Return the runtime started by the application lifespan."""
    return request.app.state.runtime


def current_investigator(
    runtime: Annotated[Runtime, Depends(current_runtime)],
) -> Investigator:
    """Return the investigator, or fail when no model is configured."""
    if runtime.investigator is None:
        raise HTTPException(503, "EXPLABS_API_KEY is not configured")
    return runtime.investigator


RuntimeDependency = Annotated[Runtime, Depends(current_runtime)]
InvestigatorDependency = Annotated[Investigator, Depends(current_investigator)]


def create_app(runtime_factory: RuntimeFactory = production_runtime) -> FastAPI:
    """Create the API; the runtime starts and stops with the application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory() as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(
        title="Engineering Change Investigator",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(InvestigationError)
    async def investigation_error(
        request: Request, error: InvestigationError
    ) -> JSONResponse:
        del request
        return JSONResponse({"detail": error.detail}, status_code=error.status)

    @app.get("/api/health")
    async def health(runtime: RuntimeDependency) -> dict[str, Any]:
        counts = await asyncio.to_thread(runtime.backends.counts)
        return {
            "status": "ok",
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
            "policy_version": policy.version,
            "default_effect": policy.default_effect.value,
            "consequential": {RELEASE_DECISION: "investigation_id"},
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

    @app.post("/api/investigate", response_model=Investigation)
    async def investigate(
        request: InvestigationRequest, investigator: InvestigatorDependency
    ) -> Investigation:
        return await investigator.investigate(request.question)

    @app.get("/api/investigations/{investigation_id}", response_model=Investigation)
    async def investigation(
        investigation_id: UUID, investigator: InvestigatorDependency
    ) -> Investigation:
        return await investigator.get(investigation_id)

    @app.post(
        "/api/investigations/{investigation_id}/approval",
        response_model=Investigation,
    )
    async def approval(
        investigation_id: UUID,
        request: ApprovalRequest,
        investigator: InvestigatorDependency,
    ) -> Investigation:
        return await investigator.review(investigation_id, request)

    @app.post(
        "/api/investigations/{investigation_id}/publication",
        response_model=Investigation,
    )
    async def publication(
        investigation_id: UUID, investigator: InvestigatorDependency
    ) -> Investigation:
        return await investigator.publish_reconciled(investigation_id)

    @app.post("/api/investigations/{investigation_id}/feedback")
    async def feedback(
        investigation_id: UUID,
        request: FeedbackRequest,
        investigator: InvestigatorDependency,
    ) -> dict[str, str]:
        return await investigator.submit_feedback(investigation_id, request)

    @app.post("/api/refresh")
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
