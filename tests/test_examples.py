"""Checks for the runnable examples in `examples/`.

Offline (always): every example compiles, documents what it shows and how to
run it, defines `main()` behind a `__main__` guard, imports only the public
`agloom` API, and never names an agent or message (several providers reject
`name` on assistant messages). Policy files parse as agentbarrier policies.

Live (opt-in): `pytest tests/test_examples.py -m network` runs every example
as a subprocess against the model configured by `EXPLABS_*` (environment or
the repository `.env`). The PostgreSQL examples also need
`AGLOOM_POSTGRES_URL`.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from agentbarrier.runtime import RuntimePolicy
from dotenv import dotenv_values

import agloom

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
EXPECTED = (
    "01_direct.py",
    "02_automatic.py",
    "03_planner.py",
    "04_react_tools.py",
    "05_supervisor.py",
    "06_pipeline.py",
    "07_reflection.py",
    "08_swarm.py",
    "09_blackboard.py",
    "10_hybrid.py",
    "11_recursion.py",
    "12_no_recursion.py",
    "13_mcp.py",
    "14_context.py",
    "15_feedback.py",
    "16_behavior.py",
    "17_refresh.py",
    "18_explainability.py",
    "19_structured_output.py",
    "20_hitl.py",
    "21_checkpoint_resume.py",
    "22_streaming.py",
    "23_observability.py",
    "24_complete_configuration.py",
    "25_loop_automatic_completion.py",
    "26_loop_multi_iteration.py",
    "27_loop_progress.py",
    "28_loop_retry.py",
    "29_loop_replan.py",
    "30_loop_refresh.py",
    "31_loop_stagnation.py",
    "32_loop_budget.py",
    "33_loop_human_intervention.py",
    "34_loop_checkpoint_resume.py",
    "35_loop_recursive.py",
    "36_loop_topology_adaptation.py",
    "37_loop_feedback.py",
    "38_loop_context.py",
    "39_loop_long_running.py",
    "40_policy_allow_deny.py",
    "41_policy_approval.py",
    "42_policy_custom_rules.py",
    "43_policy_mcp.py",
    "44_policy_postgres.py",
    "45_capability_routing.py",
    "46_deep_agents.py",
    "47_postgres_checkpointer.py",
)
HELPERS = ("_support.py", "_mcp_server.py")
POSTGRES = frozenset({"44_policy_postgres.py", "47_postgres_checkpointer.py"})
LIVE_TIMEOUT = 600


def _tree(name: str) -> ast.Module:
    path = EXAMPLES / name
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    return ast.parse(source, filename=str(path))


def test_the_example_set_matches_the_documented_names() -> None:
    present = sorted(path.name for path in EXAMPLES.glob("[0-9][0-9]_*.py"))
    assert present == list(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_example_is_documented_and_runnable(name: str) -> None:
    tree = _tree(name)
    docstring = ast.get_docstring(tree)
    assert docstring, f"{name} needs a module docstring"
    assert f"uv run python examples/{name}" in docstring, "document how to run it"
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert "main" in functions, f"{name} needs a main() function"
    assert any(_is_main_guard(node) for node in tree.body), "add a __main__ guard"


@pytest.mark.parametrize("name", EXPECTED + HELPERS)
def test_example_uses_only_the_public_agloom_api(name: str) -> None:
    public = set(agloom.__all__)
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "agloom":
                private = [a.name for a in node.names if a.name not in public]
                assert not private, f"{name} imports non-public names {private}"
            assert not node.module.startswith("agloom."), (
                f"{name} imports the submodule {node.module}; use `from agloom ...`"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("agloom."), alias.name
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "agloom" and node.attr.startswith("_"))


@pytest.mark.parametrize("name", EXPECTED + HELPERS)
def test_example_never_sets_a_message_or_agent_name(name: str) -> None:
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.Call):
            callee = node.func
            label = callee.attr if isinstance(callee, ast.Attribute) else None
            label = callee.id if isinstance(callee, ast.Name) else label
            if label in {"create_agent", "create_deep_agent", "AIMessage"}:
                keywords = {keyword.arg for keyword in node.keywords}
                assert "name" not in keywords, f"{name}: {label}(name=...)"


@pytest.mark.parametrize("name", HELPERS)
def test_helper_module_is_documented(name: str) -> None:
    assert ast.get_docstring(_tree(name))


@pytest.mark.parametrize(
    "path", sorted((EXAMPLES / "policies").glob("*.json")), ids=lambda path: path.name
)
def test_policy_file_is_a_valid_agentbarrier_policy(path: Path) -> None:
    policy = RuntimePolicy.from_file(path)
    assert policy.rules


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _configured(variable: str) -> bool:
    if os.environ.get(variable):
        return True
    env_file = ROOT / ".env"
    return env_file.is_file() and bool(dotenv_values(env_file).get(variable))


@pytest.mark.network
@pytest.mark.timeout(LIVE_TIMEOUT + 60)
@pytest.mark.parametrize("name", EXPECTED)
def test_example_runs_live(name: str, request: pytest.FixtureRequest) -> None:
    if "network" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("live examples run only when selected with -m network")
    if not _configured("EXPLABS_API_KEY"):
        pytest.skip("EXPLABS_API_KEY is not configured")
    if name in POSTGRES and not _configured("AGLOOM_POSTGRES_URL"):
        pytest.skip("AGLOOM_POSTGRES_URL is not configured")

    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=ROOT,
        env={**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=LIVE_TIMEOUT,
        check=False,
    )

    output = f"{completed.stdout}\n{completed.stderr}".strip()
    assert completed.returncode == 0, f"{name} failed:\n{output[-4000:]}"
