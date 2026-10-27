from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import tomllib
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol
from uuid import UUID, uuid4

import httpx

INTERNAL_DIRECTORY = ".agloom"


class WorkspaceViolation(ValueError):
    pass


def _ignored_workspace_entries(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if (Path(directory) / name).is_symlink()}
    ignored.update(
        name
        for name in names
        if name in {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}
    )
    return ignored


class ExecutionBoundary(Protocol):
    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict[str, object]: ...


class RemoteExecutionBoundary:
    MAX_TRANSFER_BYTES: ClassVar[int] = 10_000_000

    def __init__(self, url: str, shared_root: Path) -> None:
        self.url = url.rstrip("/")
        self.shared_root = shared_root.resolve()

    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict[str, object]:
        resolved = workspace.resolve()
        try:
            resolved.relative_to(self.shared_root)
        except ValueError as error:
            raise WorkspaceViolation(
                "workspace is outside the shared execution root"
            ) from error
        safe_workspace = SafeWorkspace(resolved)
        files = {
            relative: safe_workspace.read(relative, SafeWorkspace.MAX_TEXT_FILE_BYTES)
            for relative in safe_workspace.inventory()
            if not relative.startswith(f"{INTERNAL_DIRECTORY}/")
        }
        if (
            sum(len(value.encode("utf-8")) for value in files.values())
            > self.MAX_TRANSFER_BYTES
        ):
            raise WorkspaceViolation("workspace exceeds isolated transfer limit")
        response = httpx.post(
            f"{self.url}/execute",
            json={
                "files": files,
                "commands": [list(command) for command in commands],
                "timeout": timeout,
            },
            timeout=timeout + 10,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or "passed" not in result:
            raise WorkspaceViolation("execution boundary returned an invalid response")
        return result


class LocalExecutionBoundary:
    """Test/development boundary; production uses the isolated executor service."""

    SAFE_ENV: ClassVar[frozenset[str]] = frozenset(
        {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TMP", "TEMP"}
    )

    def __init__(
        self,
        execution_root: Path,
        *,
        frontend_modules: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.execution_root = execution_root.resolve()
        self.frontend_modules = frontend_modules
        self.python_executable = python_executable

    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict[str, object]:
        self.execution_root.mkdir(parents=True, exist_ok=True)
        execution = self.execution_root / uuid4().hex
        shutil.copytree(
            workspace,
            execution,
            symlinks=False,
            ignore=_ignored_workspace_entries,
        )
        outputs: list[str] = []
        exit_code = 0
        env = {
            name: value for name, value in os.environ.items() if name in self.SAFE_ENV
        }
        env.update(
            {
                "CI": "true",
                "HOME": str(execution),
                "NO_PROXY": "*",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            for original in commands:
                command = original
                cwd = execution / ("frontend" if original[0] == "npm" else "backend")
                if original[0] == "npm" and original[1] == "install":
                    if self.frontend_modules is None:
                        continue
                    shutil.copytree(
                        self.frontend_modules,
                        cwd / "node_modules",
                        symlinks=True,
                    )
                    continue
                if original[0] == "uv" and self.python_executable:
                    command = (self.python_executable, "-m", "pytest", "-q")
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    check=False,
                    env=env,
                )
                outputs.extend((completed.stdout, completed.stderr))
                exit_code = completed.returncode
                if exit_code:
                    break
        except subprocess.TimeoutExpired as error:
            return {
                "passed": False,
                "exit_code": None,
                "output": f"isolated check timed out after {error.timeout} seconds",
            }
        finally:
            shutil.rmtree(execution, ignore_errors=True)
        return {
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "output": "\n".join(outputs)[-12_000:],
        }

    def run_files(
        self,
        files: Mapping[str, str],
        commands: list[tuple[str, ...]],
        timeout: int,
    ) -> dict[str, object]:
        incoming = self.execution_root / f".incoming-{uuid4().hex}"
        workspace = SafeWorkspace(incoming)
        try:
            for relative, content in files.items():
                workspace.write(relative, content)
            return self.run(incoming, commands, timeout)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)


def _ignored_release_entries(directory: str, names: list[str]) -> set[str]:
    ignored = _ignored_workspace_entries(directory, names)
    if INTERNAL_DIRECTORY in names:
        ignored.add(INTERNAL_DIRECTORY)
    return ignored


def _transient_name(owner: str, purpose: str) -> str:
    """A hidden sibling name for a temporary copy, shorter than a UUID name.

    Temporary copies therefore never need longer paths than the directories
    they become (Windows limits directory paths to 248 characters).
    """
    return f".{owner[:8]}.{purpose}-{uuid4().hex[:8]}"


@dataclass
class RunWorkspace:
    """The private working copy of one run, stable across job attempts.

    A run builds in `projects/.runs/<project>/<run>`, a copy of the project's
    active workspace taken when the run first starts. Retries and resumed runs
    reopen the same directory, so a run continued from its checkpoint sees the
    files its earlier tool calls wrote. Only `publish` changes the active
    workspace.
    """

    active_root: Path
    run_root: Path
    workspace: SafeWorkspace

    @classmethod
    def open(
        cls,
        projects_root: Path,
        project_id: UUID,
        run_id: UUID,
        execution_boundary: ExecutionBoundary | None = None,
    ) -> RunWorkspace:
        projects = projects_root.resolve()
        active = projects / str(project_id)
        root = projects / ".runs" / str(project_id) / str(run_id)
        if not root.is_dir():
            staging = root.with_name(_transient_name(str(run_id), "staging"))
            staging.parent.mkdir(parents=True, exist_ok=True)
            if active.is_dir():
                shutil.copytree(
                    active, staging, symlinks=False, ignore=_ignored_workspace_entries
                )
            else:
                staging.mkdir()
            try:
                staging.replace(root)
            except OSError:
                shutil.rmtree(staging, ignore_errors=True)
                if not root.is_dir():
                    raise
        return cls(active, root, SafeWorkspace(root, execution_boundary))

    def publish(self, guard: AbstractContextManager[None]) -> list[str]:
        """Atomically replace the active workspace with this run's files.

        Internal evidence under `.agloom/` stays with the run. `guard` serializes
        publications of the same project.
        """
        files = [
            path
            for path in self.workspace.inventory()
            if not path.startswith(f"{INTERNAL_DIRECTORY}/")
        ]
        parent = self.active_root.parent
        staging = parent / _transient_name(self.active_root.name, "publishing")
        backup = parent / _transient_name(self.active_root.name, "previous")
        shutil.copytree(
            self.run_root, staging, symlinks=False, ignore=_ignored_release_entries
        )
        try:
            with guard:
                if self.active_root.exists():
                    self.active_root.replace(backup)
                try:
                    staging.replace(self.active_root)
                except BaseException:
                    if backup.exists():
                        backup.replace(self.active_root)
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
        return files

    def discard(self) -> None:
        shutil.rmtree(self.run_root, ignore_errors=True)


class SafeWorkspace:
    MAX_INVENTORY_FILES: ClassVar[int] = 500
    MAX_TEXT_FILE_BYTES: ClassVar[int] = 256_000
    EXCLUDED_DIRECTORIES: ClassVar[frozenset[str]] = frozenset(
        {
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "node_modules",
            "dist",
            "build",
            "coverage",
            ".coverage",
        }
    )
    TEXT_SUFFIXES: ClassVar[frozenset[str]] = frozenset(
        {
            ".css",
            ".csv",
            ".env",
            ".gitignore",
            ".html",
            ".ini",
            ".js",
            ".json",
            ".jsx",
            ".lock",
            ".md",
            ".mjs",
            ".py",
            ".pyi",
            ".sql",
            ".svg",
            ".toml",
            ".ts",
            ".tsx",
            ".txt",
            ".yaml",
            ".yml",
        }
    )
    FRONTEND_PACKAGES: ClassVar[frozenset[str]] = frozenset(
        {
            "react",
            "react-dom",
            "vite",
            "typescript",
            "@vitejs/plugin-react",
            "@types/react",
            "@types/react-dom",
            "vitest",
        }
    )
    FRONTEND_SCRIPTS: ClassVar[dict[str, str]] = {
        "build": "tsc && vite build",
        "test": "vitest run",
        "dev": "vite --host 0.0.0.0",
    }
    BACKEND_PACKAGES: ClassVar[frozenset[str]] = frozenset(
        {"fastapi", "httpx", "pydantic", "pytest", "pytest-asyncio"}
    )
    BACKEND_TEST: ClassVar[tuple[str, ...]] = (
        "uv",
        "run",
        "--no-project",
        "--with",
        "fastapi==0.116.1",
        "--with",
        "httpx==0.28.1",
        "--with",
        "pydantic==2.11.9",
        "--with",
        "pytest==8.4.2",
        "--with",
        "pytest-asyncio==1.2.0",
        "pytest",
        "-q",
    )
    CHECKS: ClassVar[frozenset[str]] = frozenset(
        {"frontend-build", "frontend-test", "backend-test"}
    )
    _SAFE_VERSION = re.compile(r"^[0-9A-Za-z.*+!<>=~^,_ -]+$")

    def _frontend_manifest(self) -> dict[str, object]:
        path = self.resolve("frontend/package.json")
        if not path.is_file():
            raise WorkspaceViolation("frontend/package.json is required")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise WorkspaceViolation("frontend/package.json is invalid") from error
        if not isinstance(manifest, dict):
            raise WorkspaceViolation("frontend/package.json must be an object")
        forbidden_fields = {
            "optionalDependencies",
            "peerDependencies",
            "bundledDependencies",
            "bundleDependencies",
            "overrides",
            "resolutions",
            "workspaces",
        }
        present = forbidden_fields.intersection(manifest)
        if present:
            raise WorkspaceViolation(
                f"frontend package fields are not allowed: {sorted(present)}"
            )
        for section in ("dependencies", "devDependencies"):
            values = manifest.get(section, {})
            if not isinstance(values, dict):
                raise WorkspaceViolation(f"{section} must be an object")
            for name, version in values.items():
                if name not in self.FRONTEND_PACKAGES:
                    raise WorkspaceViolation(
                        f"frontend dependency is not allowlisted: {name}"
                    )
                if not isinstance(version, str) or not self._SAFE_VERSION.fullmatch(
                    version
                ):
                    raise WorkspaceViolation(
                        f"frontend dependency has an unsafe version: {name}"
                    )
        scripts = manifest.get("scripts", {})
        if not isinstance(scripts, dict):
            raise WorkspaceViolation("scripts must be an object")
        for name, command in scripts.items():
            if self.FRONTEND_SCRIPTS.get(name) != command:
                raise WorkspaceViolation(
                    f"frontend package script is not allowlisted: {name}"
                )
        return manifest

    def _validate_backend_manifest(self) -> None:
        path = self.resolve("backend/pyproject.toml")
        if not path.is_file():
            raise WorkspaceViolation("backend/pyproject.toml is required")
        try:
            manifest = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            raise WorkspaceViolation("backend/pyproject.toml is invalid") from error
        if "build-system" in manifest:
            raise WorkspaceViolation("backend build-system is not allowed")
        project = manifest.get("project", {})
        if not isinstance(project, dict):
            raise WorkspaceViolation("backend project table is invalid")
        if project.get("scripts") or project.get("entry-points"):
            raise WorkspaceViolation("backend package scripts are not allowed")
        dependency_groups = manifest.get("dependency-groups", {})
        if dependency_groups and not isinstance(dependency_groups, dict):
            raise WorkspaceViolation("backend dependency-groups are invalid")
        project_dependencies = project.get("dependencies", [])
        if not isinstance(project_dependencies, list):
            raise WorkspaceViolation("backend dependencies must be a list")
        dependencies = list(project_dependencies)
        if isinstance(dependency_groups, dict):
            for values in dependency_groups.values():
                if not isinstance(values, list):
                    raise WorkspaceViolation("backend dependency group must be a list")
                dependencies.extend(values)
        optional = project.get("optional-dependencies", {})
        if optional:
            raise WorkspaceViolation("backend optional dependencies are not allowed")
        for dependency in dependencies:
            if not isinstance(dependency, str):
                raise WorkspaceViolation("backend dependency must be a string")
            match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", dependency)
            name = match.group(1).lower().replace("_", "-") if match else ""
            suffix = dependency[len(match.group(0)) :] if match else dependency
            if name not in self.BACKEND_PACKAGES:
                raise WorkspaceViolation(
                    f"backend dependency is not allowlisted: {name or dependency}"
                )
            if (
                not suffix
                or not self._SAFE_VERSION.fullmatch(suffix)
                or any(marker in dependency for marker in ("@", ";", "/", "\\", ":"))
            ):
                raise WorkspaceViolation(
                    f"backend dependency has an unsafe version: {dependency}"
                )

    def __init__(
        self, root: Path, execution_boundary: ExecutionBoundary | None = None
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_boundary = execution_boundary
        self._check_results: dict[tuple[str, str], dict[str, object]] = {}
        self._check_lock = threading.Lock()

    def resolve(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise WorkspaceViolation("path must be non-empty and relative")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceViolation("path escapes the project workspace") from error
        return candidate

    def read(self, relative: str, limit: int = 50_000) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        if not self._is_inventory_file(path):
            raise WorkspaceViolation("file is not an allowed bounded text artifact")
        bounded = min(max(limit, 1), self.MAX_TEXT_FILE_BYTES)
        try:
            with path.open("r", encoding="utf-8") as stream:
                return stream.read(bounded)
        except UnicodeDecodeError as error:
            raise WorkspaceViolation("file is not valid UTF-8 text") from error

    def write(self, relative: str, content: str) -> Path:
        path = self.resolve(relative)
        if INTERNAL_DIRECTORY in path.relative_to(self.root).parts:
            raise WorkspaceViolation(".agloom is reserved for internal evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_playwright_evidence(self, payload: Mapping[str, Any]) -> Path:
        """Persist trusted browser evidence to the single internal marker."""
        required = {"navigation_succeeded", "url", "snapshot"}
        if set(payload) != required:
            raise WorkspaceViolation("Playwright evidence has an invalid shape")
        if (
            not isinstance(payload["navigation_succeeded"], bool)
            or not isinstance(payload["url"], str)
            or not isinstance(payload["snapshot"], str)
        ):
            raise WorkspaceViolation("Playwright evidence has invalid field types")
        return self._write_internal("playwright.json", dict(payload))

    def write_release_marker(self, payload: Mapping[str, Any]) -> Path:
        """Record the release this run published (trusted application code only)."""
        return self._write_internal("release.json", dict(payload))

    def read_release_marker(self) -> dict[str, Any] | None:
        path = self.resolve(f"{INTERNAL_DIRECTORY}/release.json")
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_internal(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.resolve(f"{INTERNAL_DIRECTORY}/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def fingerprint(self) -> str:
        """Digest of every source file, excluding internal evidence."""
        digest = hashlib.sha256()
        for relative in self.inventory():
            if relative.startswith(f"{INTERNAL_DIRECTORY}/"):
                continue
            content = self.resolve(relative).read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).digest())
        return digest.hexdigest()

    def inventory(self) -> list[str]:
        files: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not self._is_inventory_file(path):
                continue
            files.append(str(path.relative_to(self.root)).replace("\\", "/"))
            if len(files) >= self.MAX_INVENTORY_FILES:
                break
        return files

    def _is_inventory_file(self, path: Path) -> bool:
        if not path.is_file() or path.is_symlink():
            return False
        relative = path.relative_to(self.root)
        if any(part in self.EXCLUDED_DIRECTORIES for part in relative.parts[:-1]):
            return False
        if path.stat().st_size > self.MAX_TEXT_FILE_BYTES:
            return False
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in self.TEXT_SUFFIXES and name not in {
            "dockerfile",
            "containerfile",
        }:
            return False
        try:
            with path.open("rb") as stream:
                sample = stream.read(8_192)
            sample.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        return b"\x00" not in sample

    def run_check(self, name: str, timeout: int = 120) -> dict[str, object]:
        if name not in self.CHECKS:
            raise WorkspaceViolation(f"check is not allowlisted: {name}")
        cwd = self.root / ("frontend" if name.startswith("frontend") else "backend")
        if not cwd.is_dir():
            return {
                "passed": False,
                "exit_code": None,
                "output": f"missing {cwd.name}/",
            }
        commands: list[tuple[str, ...]]
        if name.startswith("frontend"):
            manifest = self._frontend_manifest()
            script = "build" if name == "frontend-build" else "test"
            if script not in manifest.get("scripts", {}):
                raise WorkspaceViolation(
                    f"frontend package script is required: {script}"
                )
            node_modules = self.resolve("frontend/node_modules")
            if node_modules.exists():
                if node_modules.is_symlink():
                    raise WorkspaceViolation(
                        "frontend/node_modules cannot be a symlink"
                    )
                shutil.rmtree(node_modules)
            commands = [
                (
                    "npm",
                    "install",
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                    "--package-lock=false",
                ),
                ("npm", "run", script),
            ]
        else:
            self._validate_backend_manifest()
            commands = [self.BACKEND_TEST]
        if self.execution_boundary is None:
            raise WorkspaceViolation("an isolated execution boundary is required")
        key = (name, self.fingerprint())
        with self._check_lock:
            cached = self._check_results.get(key)
        if cached is not None:
            return dict(cached)
        result = self.execution_boundary.run(
            self.root, commands, min(max(timeout, 1), 300)
        )
        with self._check_lock:
            self._check_results[key] = dict(result)
        return result
