from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .safety import LocalExecutionBoundary, SafeWorkspace

EXECUTIONS = Path("/validation-runs")
FRONTEND_MODULES = Path("/opt/agloom-validator/frontend/node_modules")
PYTHON = "/opt/venv/bin/python"

FRONTEND_INSTALL = (
    "npm",
    "install",
    "--ignore-scripts",
    "--no-audit",
    "--no-fund",
    "--package-lock=false",
)
ALLOWED_COMMANDS = {
    FRONTEND_INSTALL,
    ("npm", "run", "build"),
    ("npm", "run", "test"),
}


class ExecutionRequest(BaseModel):
    files: dict[str, str] = Field(max_length=500)
    commands: list[list[str]] = Field(min_length=1, max_length=2)
    timeout: int = Field(ge=1, le=300)


app = FastAPI(
    title="Agloom isolated validation executor", docs_url=None, redoc_url=None
)
boundary = LocalExecutionBoundary(
    EXECUTIONS,
    frontend_modules=FRONTEND_MODULES,
    python_executable=PYTHON,
)
execution_lock = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute")
def execute(request: ExecutionRequest) -> dict[str, object]:
    if sum(len(value.encode("utf-8")) for value in request.files.values()) > 10_000_000:
        raise HTTPException(413, "workspace exceeds isolated transfer limit")
    commands = [tuple(command) for command in request.commands]
    backend_command = commands == [SafeWorkspace.BACKEND_TEST]
    frontend_commands = (
        len(commands) == 2
        and commands[0] == FRONTEND_INSTALL
        and commands[1] in ALLOWED_COMMANDS
    )
    if not (backend_command or frontend_commands):
        raise HTTPException(400, "command pipeline is not allowlisted")
    try:
        with execution_lock:
            return boundary.run_files(request.files, commands, request.timeout)
    except (OSError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
