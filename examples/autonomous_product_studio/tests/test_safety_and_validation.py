from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
from agloom import (
    DefaultLoopController,
    DefaultLoopEvaluator,
    LoopAction,
    LoopBudget,
    LoopPolicy,
    LoopState,
)
from fastapi import HTTPException

from backend import validation_executor
from backend.looping import ValidationLoopObserver
from backend.models import AcceptanceCriterion, LoopRun, RunProof
from backend.safety import (
    LocalExecutionBoundary,
    RunWorkspace,
    SafeWorkspace,
    WorkspaceViolation,
)
from backend.store import MemoryStudioStore
from backend.validation import (
    ObjectiveValidator,
    release_is_current,
    validation_feedback,
)
from tests.conftest import TEST_PRINCIPAL
from tests.support import RecordingBoundary


def tracked_run(store: MemoryStudioStore) -> UUID:
    run = LoopRun(
        id=uuid4(),
        task_id=uuid4(),
        status="running",
        proof=RunProof(),
        created_at="2026-01-01T00:00:00Z",
    )
    store.runs[run.id] = run
    store.owners[run.id] = TEST_PRINCIPAL.tenant_id
    return run.id


def observer_for(
    workspace: SafeWorkspace,
    criteria: list[AcceptanceCriterion],
    store: MemoryStudioStore | None = None,
    **options,
) -> ValidationLoopObserver:
    studio = store or MemoryStudioStore()
    return ValidationLoopObserver(
        ObjectiveValidator(workspace),
        studio,
        tracked_run(studio),
        criteria,
        workspace,
        **options,
    )


def publish_marker(workspace: SafeWorkspace) -> None:
    workspace.write_release_marker({"fingerprint": workspace.fingerprint()})


def test_workspace_rejects_absolute_and_traversal_paths(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    with pytest.raises(WorkspaceViolation):
        workspace.resolve("../outside.txt")
    with pytest.raises(WorkspaceViolation):
        workspace.resolve(str((tmp_path / "absolute.txt").resolve()))
    with pytest.raises(WorkspaceViolation):
        workspace.run_check("rm-everything")


def _decide(observation):
    policy = LoopPolicy(max_iterations=3)
    budget = LoopBudget.from_policy(policy).model_copy(update={"iterations": 1})
    state = LoopState(loop_id="loop", objective="Build", budget=budget)
    evaluation = DefaultLoopEvaluator().evaluate(state, observation, policy)
    return DefaultLoopController().decide(
        state, observation, evaluation, policy, budget
    )


APP_CRITERION = AcceptanceCriterion(
    id="AC-1",
    description="React app exists",
    kind="file",
    target="frontend/src/App.tsx",
)


def test_validated_and_published_release_completes_the_loop(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    store = MemoryStudioStore()
    observer = observer_for(workspace, [APP_CRITERION], store)
    publish_marker(workspace)

    observation = observer.observe(iteration=1, result={"messages": []})
    decision = _decide(observation)

    assert observation.validation_passed is True
    assert observation.objective_progress == 1
    assert observation.metadata["release_published"] is True
    assert decision.action is LoopAction.COMPLETE
    assert store.evidence_for_run(observer.run_id)[0].passed is True


def test_passing_criteria_without_a_release_keep_the_loop_going(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    observer = observer_for(workspace, [APP_CRITERION])

    observation = observer.observe(iteration=1, result="built, not published")
    decision = _decide(observation)

    assert observation.validation_passed is False
    assert observation.objective_progress == 0.5
    assert observation.quality == 1
    assert decision.action is LoopAction.CONTINUE


def test_a_stale_release_does_not_complete_the_objective(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    publish_marker(workspace)
    workspace.write(
        "frontend/src/App.tsx", "export default function App() { return 1 }"
    )

    assert release_is_current(workspace) is False
    assert (
        observer_for(workspace, [APP_CRITERION]).observe(iteration=1).validation_passed
        is False
    )


def test_incomplete_validation_continues_within_budget(tmp_path):
    criteria = [
        AcceptanceCriterion(
            id="AC-1",
            description="Backend exists",
            kind="file",
            target="backend/app.py",
        )
    ]
    observer = observer_for(SafeWorkspace(tmp_path / "workspace"), criteria)
    observation = observer.observe(iteration=1, result="work in progress")
    assert observation.validation_passed is False
    assert _decide(observation).action is LoopAction.CONTINUE


def test_manifest_policy_violation_becomes_repairable_loop_feedback(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write(
        "frontend/package.json",
        json.dumps(
            {
                "scripts": {"build": "vite build"},
                "dependencies": {"react": "^19.1.1", "lucide-react": "latest"},
            }
        ),
    )
    criterion = AcceptanceCriterion(
        id="AC-1",
        description="Frontend production build succeeds",
        kind="command",
        target="frontend-build",
    )
    store = MemoryStudioStore()
    observer = observer_for(workspace, [criterion], store)

    observation = observer.observe(iteration=1, result="first attempt")
    feedback = validation_feedback(
        store.evidence_for_run(observer.run_id), released=False
    )

    assert observation.validation_passed is False
    assert observation.retryable is True
    assert feedback["failed_criteria"] == ["AC-1"]
    assert (
        feedback["results"][0]["details"]["error"]
        == "frontend dependency is not allowlisted: lucide-react"
    )


def test_playwright_marker_is_consumed_as_objective_evidence(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write_playwright_evidence(
        {
            "navigation_succeeded": True,
            "url": "http://target-preview",
            "snapshot": "Product preview ready",
        }
    )
    criterion = AcceptanceCriterion(
        id="AC-4",
        description="Preview is browser validated",
        kind="browser",
        target="http://target-preview",
        expected="Product preview ready",
    )
    result = ObjectiveValidator(workspace).validate([criterion])[0]
    assert result.source == "playwright"
    assert result.passed is True


@pytest.mark.parametrize(
    "payload",
    [
        {
            "navigation_succeeded": True,
            "url": "http://attacker.invalid",
            "snapshot": "Product preview ready",
        },
        {
            "navigation_succeeded": True,
            "url": "http://target-preview",
            "snapshot": "Wrong page",
        },
        {
            "navigation_succeeded": False,
            "url": "http://target-preview",
            "snapshot": "Product preview ready",
        },
    ],
)
def test_browser_validation_rejects_wrong_url_content_or_navigation(tmp_path, payload):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write_playwright_evidence(payload)
    criterion = AcceptanceCriterion(
        id="AC-4",
        description="Preview is browser validated",
        kind="browser",
        target="http://target-preview",
        expected="Product preview ready",
    )
    assert ObjectiveValidator(workspace).validate([criterion])[0].passed is False


def test_feedback_reports_the_latest_iteration_and_the_release_step(tmp_path):
    criterion = AcceptanceCriterion(
        id="AC-1",
        description="Backend exists",
        kind="file",
        target="backend/app.py",
    )
    workspace = SafeWorkspace(tmp_path / "workspace")
    store = MemoryStudioStore()
    observer = observer_for(workspace, [criterion], store)
    assert (
        validation_feedback([], released=False)["instruction"]
        == "No validation has run yet."
    )

    observer.observe(iteration=1, result="first attempt")
    first = validation_feedback(store.evidence_for_run(observer.run_id), released=False)
    workspace.write("backend/app.py", "app = None")
    observer.observe(iteration=2, result="second attempt")
    second = validation_feedback(
        store.evidence_for_run(observer.run_id), released=False
    )
    published = validation_feedback(
        store.evidence_for_run(observer.run_id), released=True
    )

    assert (first["iteration"], first["failed_criteria"]) == (1, ["AC-1"])
    assert first["results"][0]["passed"] is False
    assert (second["iteration"], second["failed_criteria"]) == (2, [])
    assert "publish_release" in second["instruction"]
    assert (
        published["instruction"] == "All criteria passed and the release is published."
    )


def test_resumed_iterations_overwrite_rather_than_duplicate_evidence(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    store = MemoryStudioStore()
    observer = observer_for(workspace, [APP_CRITERION], store)
    observer.observe(iteration=1, result="interrupted")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    observer.observe(iteration=1, result="replayed")

    [evidence] = store.evidence_for_run(observer.run_id)
    assert evidence.passed is True


def test_observer_streams_deterministic_criterion_progress(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    events = []
    criteria = [
        APP_CRITERION,
        AcceptanceCriterion(
            id="AC-2",
            description="Backend exists",
            kind="file",
            target="backend/app.py",
        ),
    ]
    observer = observer_for(workspace, criteria, on_iteration=events.append)

    observer.observe(iteration=2, result="partial implementation")

    assert events == [
        {
            "iteration": 2,
            "progress": 1 / 3,
            "quality": 0.5,
            "passed_criteria": 1,
            "acceptance_criteria": 2,
            "failed_criteria": ["AC-2"],
            "release_published": False,
            "message": "1 of 2 acceptance criteria passed.",
        }
    ]


def test_allowed_frontend_installs_without_scripts_then_builds(tmp_path, monkeypatch):
    calls = []

    class Boundary:
        def run(self, root, commands, timeout):
            calls.append((root, commands, timeout))
            return {"passed": True, "exit_code": 0, "output": "ok"}

    workspace = SafeWorkspace(tmp_path / "workspace", Boundary())
    workspace.write(
        "frontend/package.json",
        json.dumps(
            {
                "scripts": {"build": "tsc && vite build"},
                "dependencies": {"react": "^19.1.1", "react-dom": "^19.1.1"},
                "devDependencies": {"typescript": "^5.9.2", "vite": "^7.1.7"},
            }
        ),
    )
    result = workspace.run_check("frontend-build")
    assert result["passed"] is True
    assert calls[0][1][0] == (
        "npm",
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--package-lock=false",
    )
    assert calls[0][1][1] == ("npm", "run", "build")


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            {
                "scripts": {"build": "tsc && vite build"},
                "dependencies": {"left-pad": "1.3.0"},
            },
            "dependency is not allowlisted",
        ),
        (
            {
                "scripts": {
                    "build": "tsc && vite build",
                    "postinstall": "node steal.js",
                },
                "dependencies": {"react": "^19.1.1"},
            },
            "package script is not allowlisted",
        ),
        (
            {
                "scripts": {"build": "node steal.js"},
                "dependencies": {"react": "^19.1.1"},
            },
            "package script is not allowlisted",
        ),
        (
            {
                "scripts": {"build": "tsc && vite build"},
                "dependencies": {"react": "^19.1.1"},
                "optionalDependencies": {"left-pad": "1.3.0"},
            },
            "package fields are not allowed",
        ),
    ],
)
def test_frontend_rejects_untrusted_dependencies_and_scripts(
    tmp_path, manifest, message
):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/package.json", json.dumps(manifest))
    with pytest.raises(WorkspaceViolation, match=message):
        workspace.run_check("frontend-build")


def test_allowed_backend_uses_fixed_no_project_environment(tmp_path, monkeypatch):
    calls = []

    class Boundary:
        def run(self, root, commands, timeout):
            calls.append((root, commands, timeout))
            return {"passed": True, "exit_code": 0, "output": "9 passed"}

    workspace = SafeWorkspace(tmp_path / "workspace", Boundary())
    workspace.write(
        "backend/pyproject.toml",
        """
[project]
name = "generated-api"
version = "0.1.0"
dependencies = ["fastapi>=0.116,<1", "pydantic>=2,<3"]

[dependency-groups]
dev = ["pytest>=8,<10", "pytest-asyncio>=1,<2"]
""".strip(),
    )
    result = workspace.run_check("backend-test")
    assert result["passed"] is True
    assert calls[0][1] == [SafeWorkspace.BACKEND_TEST]
    assert "--no-project" in calls[0][1][0]


def test_check_requires_an_execution_boundary(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write(
        "frontend/package.json",
        json.dumps(
            {
                "scripts": {"build": "tsc && vite build"},
                "dependencies": {"react": "19.1.1"},
            }
        ),
    )
    with pytest.raises(WorkspaceViolation, match="isolated execution boundary"):
        workspace.run_check("frontend-build")


def test_local_boundary_uses_disposable_copy_and_scrubs_credentials(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "attempt"
    (workspace / "backend").mkdir(parents=True)
    (workspace / "backend" / "test_app.py").write_text("def test_ok(): pass")
    observed = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        observed["command"] = command
        assert kwargs["cwd"] != workspace / "backend"
        assert (kwargs["cwd"] / "test_app.py").is_file()
        assert "DATABASE_URL" not in kwargs["env"]
        assert "EXPLABS_API_KEY" not in kwargs["env"]

        class Result:
            returncode = 0
            stdout = "passed"
            stderr = ""

        return Result()

    monkeypatch.setenv("DATABASE_URL", "secret")
    monkeypatch.setenv("EXPLABS_API_KEY", "secret")
    monkeypatch.setattr("subprocess.run", run)
    boundary = LocalExecutionBoundary(
        tmp_path / "executions", python_executable="python"
    )
    assert boundary.run(workspace, [SafeWorkspace.BACKEND_TEST], 10)["passed"]
    assert list((tmp_path / "executions").iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="npm executable links require POSIX")
def test_preinstalled_frontend_executables_keep_relative_symlinks(
    tmp_path, monkeypatch
):
    modules = tmp_path / "trusted-modules"
    binary = modules / ".bin"
    library = modules / "typescript" / "bin"
    binary.mkdir(parents=True)
    library.mkdir(parents=True)
    (library / "tsc").write_text("trusted")
    (binary / "tsc").symlink_to("../typescript/bin/tsc")

    def run(command, **kwargs):
        executable = kwargs["cwd"] / "node_modules" / ".bin" / "tsc"
        assert executable.is_symlink()
        assert executable.resolve().read_text() == "trusted"

        class Result:
            returncode = 0
            stdout = "passed"
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", run)
    boundary = LocalExecutionBoundary(tmp_path / "runs", frontend_modules=modules)
    result = boundary.run_files(
        {"frontend/package.json": "{}"},
        [validation_executor.FRONTEND_INSTALL, ("npm", "run", "build")],
        10,
    )
    assert result["passed"] is True
    assert list((tmp_path / "runs").iterdir()) == []


def test_run_workspace_is_reused_by_later_attempts(tmp_path):
    projects = tmp_path / "projects"
    project_id, run_id = uuid4(), uuid4()
    SafeWorkspace(projects / str(project_id)).write("app.txt", "active")
    first = RunWorkspace.open(projects, project_id, run_id)
    first.workspace.write("app.txt", "in progress")
    second = RunWorkspace.open(projects, project_id, run_id)

    assert second.run_root == first.run_root
    assert second.workspace.read("app.txt") == "in progress"
    assert SafeWorkspace(projects / str(project_id)).read("app.txt") == "active"


def test_publication_is_guarded_atomic_and_excludes_internal_evidence(tmp_path):
    projects = tmp_path / "projects"
    project_id = uuid4()
    active = projects / str(project_id)
    SafeWorkspace(active).write("app.txt", "active")
    attempt = RunWorkspace.open(projects, project_id, uuid4())
    attempt.workspace.write("app.txt", "release")
    attempt.workspace.write_release_marker({"fingerprint": "x"})

    @contextmanager
    def rejected_guard():
        raise RuntimeError("publication refused")
        yield

    with pytest.raises(RuntimeError, match="publication refused"):
        attempt.publish(rejected_guard())
    assert SafeWorkspace(active).read("app.txt") == "active"

    @contextmanager
    def accepted_guard():
        yield

    files = attempt.publish(accepted_guard())
    assert files == ["app.txt"]
    assert SafeWorkspace(active).read("app.txt") == "release"
    assert not (active / ".agloom").exists()
    assert attempt.workspace.read("app.txt") == "release"
    assert sorted(path.name for path in projects.iterdir()) == [
        ".runs",
        str(project_id),
    ]


def test_temporary_copies_are_shorter_than_their_final_directories(
    tmp_path, monkeypatch
):
    # Windows limits directory paths to 248 characters; temporary copies must
    # never be the first place a deep workspace path hits that limit.
    destinations: list[str] = []
    copytree = shutil.copytree

    def recording_copytree(source, destination, *args, **kwargs):
        destinations.append(os.path.basename(os.fspath(destination)))
        return copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", recording_copytree)
    projects = tmp_path / "projects"
    project_id, run_id = uuid4(), uuid4()
    SafeWorkspace(projects / str(project_id)).write("app.txt", "active")
    attempt = RunWorkspace.open(projects, project_id, run_id)
    attempt.workspace.write("app.txt", "release")

    @contextmanager
    def accepted_guard():
        yield

    attempt.publish(accepted_guard())

    assert len(destinations) == 2
    for name in destinations:
        assert name.startswith(".")
        assert len(name) < len(str(run_id))
    assert SafeWorkspace(projects / str(project_id)).read("app.txt") == "release"
    attempt.discard()
    assert not attempt.run_root.exists()


def test_fingerprint_tracks_sources_but_not_internal_evidence(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("app.py", "print(1)")
    before = workspace.fingerprint()
    workspace.write_playwright_evidence(
        {"navigation_succeeded": True, "url": "u", "snapshot": "s"}
    )
    workspace.write_release_marker({"fingerprint": before})

    assert workspace.fingerprint() == before
    workspace.write("app.py", "print(2)")
    assert workspace.fingerprint() != before


def test_identical_sources_reuse_a_previous_check_result(tmp_path):
    boundary = RecordingBoundary()
    workspace = SafeWorkspace(tmp_path / "workspace", boundary)
    workspace.write(
        "backend/pyproject.toml",
        '[project]\nname = "api"\nversion = "0.1.0"\ndependencies = ["fastapi>=0.116,<1"]\n',
    )
    assert workspace.run_check("backend-test")["passed"] is True
    assert workspace.run_check("backend-test")["passed"] is True
    assert len(boundary.calls) == 1
    workspace.write("backend/test_app.py", "def test_ok(): pass")
    workspace.run_check("backend-test")
    assert len(boundary.calls) == 2


def test_validation_executor_rejects_non_allowlisted_commands():
    request = validation_executor.ExecutionRequest(
        files={"backend/test_app.py": "def test_ok(): pass"},
        commands=[["python", "-c", "print('not allowed')"]],
        timeout=10,
    )
    with pytest.raises(HTTPException, match="command pipeline is not allowlisted"):
        validation_executor.execute(request)


def test_validation_executor_rejects_workspace_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation_executor,
        "boundary",
        LocalExecutionBoundary(tmp_path / "executions", python_executable="python"),
    )
    request = validation_executor.ExecutionRequest(
        files={"../outside.py": "secret = True"},
        commands=[list(SafeWorkspace.BACKEND_TEST)],
        timeout=10,
    )
    with pytest.raises(HTTPException, match="path escapes"):
        validation_executor.execute(request)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            """
[project]
name = "bad"
version = "0.1.0"
dependencies = ["requests>=2"]
""",
            "dependency is not allowlisted",
        ),
        (
            """
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "bad"
version = "0.1.0"
dependencies = ["fastapi>=0.116"]
""",
            "build-system is not allowed",
        ),
        (
            """
[project]
name = "bad"
version = "0.1.0"
dependencies = ["fastapi @ https://attacker.invalid/package.whl"]
""",
            "unsafe version",
        ),
    ],
)
def test_backend_rejects_untrusted_dependencies_and_build_hooks(
    tmp_path, manifest, message
):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("backend/pyproject.toml", manifest.strip())
    with pytest.raises(WorkspaceViolation, match=message):
        workspace.run_check("backend-test")


def test_inventory_excludes_build_caches_virtualenvs_and_binary_files(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("frontend/src/App.tsx", "export default function App() {}")
    workspace.write("frontend/dist/index.js", "compiled")
    workspace.write("backend/.venv/site.py", "dependency")
    workspace.write("backend/__pycache__/cached.py", "cache")
    workspace.resolve("frontend/logo.png").parent.mkdir(parents=True, exist_ok=True)
    workspace.resolve("frontend/logo.png").write_bytes(b"\x89PNG\x00binary")
    assert workspace.inventory() == ["frontend/src/App.tsx"]
    with pytest.raises(WorkspaceViolation, match="bounded text artifact"):
        workspace.read("frontend/logo.png")


def test_inventory_enforces_file_size_and_count_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(SafeWorkspace, "MAX_INVENTORY_FILES", 2)
    monkeypatch.setattr(SafeWorkspace, "MAX_TEXT_FILE_BYTES", 12)
    workspace = SafeWorkspace(tmp_path / "workspace")
    workspace.write("a.txt", "small")
    workspace.write("b.py", "print('b')")
    workspace.write("c.md", "third")
    workspace.write("large.txt", "x" * 13)
    inventory = workspace.inventory()
    assert len(inventory) == 2
    assert "large.txt" not in inventory
    with pytest.raises(WorkspaceViolation, match="bounded text artifact"):
        workspace.read("large.txt")


def test_inventory_skips_invalid_utf8_even_with_text_suffix(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    path = workspace.resolve("bad.txt")
    path.write_bytes(b"\xff\xfe\x00")
    assert workspace.inventory() == []


def test_public_write_cannot_forge_internal_browser_evidence(tmp_path):
    workspace = SafeWorkspace(tmp_path / "workspace")
    forged = json.dumps(
        {
            "navigation_succeeded": True,
            "url": "http://target-preview",
            "snapshot": "Product preview ready",
        }
    )
    with pytest.raises(WorkspaceViolation, match="reserved for internal evidence"):
        workspace.write(".agloom/playwright.json", forged)
    assert not workspace.resolve(".agloom/playwright.json").exists()

    workspace.write_playwright_evidence(json.loads(forged))
    persisted = json.loads(workspace.resolve(".agloom/playwright.json").read_text())
    assert persisted["navigation_succeeded"] is True
    assert persisted["url"] == "http://target-preview"
