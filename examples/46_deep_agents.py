"""Deep Agents and Agloom, in both directions.

1. Deep Agents middleware inside an Agloom agent: pass
   `deepagents.FilesystemMiddleware` in `middleware=`; its file tools (`ls`,
   `read_file`, `write_file`, `edit_file`, `delete`, `execute`, ...) become
   agent capabilities and run under Agloom's routing, policy, and loop
   budgets. Like any tool they need a policy rule: here reading is allowed,
   writing only under `/reports/`, and every other tool (editing, deleting,
   shell execution) is denied and therefore hidden from the model. The files
   live in a temporary workspace through `FilesystemBackend(virtual_mode=True)`.
2. An Agloom agent as a Deep Agents subagent: any Agloom agent is a runnable
   with `messages` state, so it plugs in as a `CompiledSubAgent`. Deep Agents
   builds its declarative subagents (including the default
   `general-purpose` one) with LangChain's `create_agent(name=...)`, which
   writes `name` on every assistant message; providers that reject `name` on
   assistant messages then fail. Registering the Agloom agent as
   `general-purpose` replaces that default, and Agloom never sets `name`.

Run:
    uv run python examples/46_deep_agents.py
"""

import tempfile
from pathlib import Path

from _support import real_model, verified_text
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from deepagents import CompiledSubAgent, FilesystemMiddleware, create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import UnsupportedContentMiddleware

from agloom import AgentEvent, EventName, PolicyManager, create_agent

TICKETS = {"T-42": "escalated to the payments team; ETA for a fix is Friday"}

WORKSPACE_POLICY = RuntimePolicy(
    version="workspace-2026-10",
    rules=(
        PolicyRule("browse the workspace", PolicyEffect.ALLOW, tool="ls"),
        PolicyRule("read files", PolicyEffect.ALLOW, tool="read_file"),
        PolicyRule(
            "write reports only",
            PolicyEffect.ALLOW,
            tool="write_file",
            conditions=(
                ArgumentCondition(
                    "file_path", ConditionOperator.STARTS_WITH, "/reports/"
                ),
            ),
        ),
    ),
)


def ticket_status(ticket_id: str) -> str:
    """Return the current status of a support ticket."""
    return f"Ticket {ticket_id}: {TICKETS.get(ticket_id, 'not found')}."


def deep_agents_middleware_in_agloom() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        workspace = Path(folder) / "workspace"
        workspace.mkdir()
        store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
        agent = create_agent(
            model=real_model(),
            pattern="react",
            middleware=[
                FilesystemMiddleware(
                    backend=FilesystemBackend(root_dir=workspace, virtual_mode=True)
                ),
                UnsupportedContentMiddleware(),
            ],
            policy=PolicyManager(policy=WORKSPACE_POLICY, store=store),
        )
        offered = set(agent.capabilities.names)
        assert {"ls", "read_file", "write_file"} <= offered, offered
        assert not offered & {"edit_file", "delete", "execute"}, offered
        assert agent.capabilities.descriptor("write_file").provider == "middleware"

        agent.invoke(
            "Write the file /reports/inc-7.md with two lines: 'Status: mitigated' "
            "and 'Owner: payments team'. Then read it back."
        )
        report = (workspace / "reports" / "inc-7.md").read_text(encoding="utf-8")
        print("Workspace file:", verified_text(report, "mitigated", "payments"))
        print("File tools offered under the policy:", ", ".join(sorted(offered)))
        store.close()


def agloom_as_deep_agents_subagent() -> None:
    model = real_model()
    events: list[AgentEvent] = []
    support = create_agent(
        model=model, tools=[ticket_status], pattern="react", observers=[events.append]
    )
    deep_agent = create_deep_agent(
        model=model,
        subagents=[
            CompiledSubAgent(
                name="general-purpose",
                description="Support specialist with access to the ticket system.",
                runnable=support,
            )
        ],
        system_prompt="Delegate every ticket question to the general-purpose "
        "subagent with the task tool, then answer the user.",
    )

    result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": "What is the status of T-42?"}]}
    )

    assert EventName.EXECUTION_COMPLETED in [e.name for e in events], "delegated"
    print("Deep agent:", verified_text(result["messages"][-1], "payments", "Friday"))


def main() -> None:
    deep_agents_middleware_in_agloom()
    agloom_as_deep_agents_subagent()


if __name__ == "__main__":
    main()
