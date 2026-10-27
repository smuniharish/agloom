# Human-in-the-loop and resume

Agloom runs pause durably and resume exactly where they stopped. This guide
covers checkpointing, the kinds of pauses, tool-call review, resuming,
conversation memory, agents inside another graph, and recovery after a crash.

## Enable checkpointing

Pauses and resumes need a LangGraph checkpointer and a
`checkpoint_authorizer`. The authorizer receives the `thread_id` of every
invocation, stream, resume, and state read, and returns whether the current
caller may use it:

```python
from langgraph.checkpoint.memory import InMemorySaver

from agloom import create_agent


def authorize(thread_id: str) -> bool:
    return thread_id.startswith("case-")  # check ownership in a real application


agent = create_agent(
    model=model,
    tools=[order_status],
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=authorize,
)
config = {"configurable": {"thread_id": "case-17"}}
```

Use `InMemorySaver` for development, `SqliteSaver` (`agloom[sqlite]`) for a
single node, and `PostgresSaver` or `AsyncPostgresSaver` (`agloom[postgres]`)
in production. Every invocation then needs `configurable.thread_id`; an
unauthorized thread raises [`ConfigurationError`][agloom.ConfigurationError]
before anything runs.

## Conversation memory

With a checkpointer, each thread keeps its conversation. Send only the new user
message on every turn:

```python
agent.invoke("My order number is A-1002.", config)
answer = agent.invoke("When will it arrive?", config)
print(answer.content)
```

```text
Your order **A-1002** is in transit and is expected to arrive on **8 October**.
```

## Kinds of pauses

| Pause | Caused by | Interrupt value | Resume with |
|---|---|---|---|
| Approval | A consequential tool call waiting for a reviewer | `type` [`APPROVAL_INTERRUPT`][agloom.APPROVAL_INTERRUPT] | `agent.resume(config)` after the reviewer decided in agentbarrier |
| Reconciliation | A consequential tool that raised, so its outcome is unknown | `type` [`RECONCILIATION_INTERRUPT`][agloom.RECONCILIATION_INTERRUPT] | `agent.resume(config)` after an operator reconciled the action in agentbarrier |
| Tool-call review | LangChain's `HumanInTheLoopMiddleware` in `middleware=` | `action_requests` and `review_configs` | `agent.resume(config, {"decisions": [...]})` |
| Breakpoint | `interrupt_before` or `interrupt_after` on a topology node | `type` [`BREAKPOINT_INTERRUPT`][agloom.BREAKPOINT_INTERRUPT] | `agent.resume(config)` |
| Loop pause | Loop Engineering waiting for a human | `type` [`LOOP_INTERRUPT`][agloom.LOOP_INTERRUPT] | `agent.resume(config, "feedback text")` |

`invoke` raises [`ExecutionPaused`][agloom.ExecutionPaused] with the pending
`interrupts` and the `thread_id`. `invoke_loop` returns the same information
as data: `result.paused` and `result.interrupts`. Branch on the interrupt
value to route each pause to the right people, for example approvals to
reviewers and reconciliations to operators:

```python
from agloom import APPROVAL_INTERRUPT, RECONCILIATION_INTERRUPT, ExecutionPaused


def route(paused: ExecutionPaused) -> str:
    value = paused.interrupts[0].value
    kind = value.get("type") if isinstance(value, dict) else None
    if kind == APPROVAL_INTERRUPT:
        return "reviewers"
    if kind == RECONCILIATION_INTERRUPT:
        return "operators"
    return "developers"  # tool-call reviews, breakpoints, and loop pauses
```

### Approvals

See [Policies and approvals](../concepts/policies.md#approvals) for the full
flow: the run pauses, a reviewer approves or rejects the action in
agentbarrier, and `agent.resume(config)` executes an approved call exactly
once.

### Reconciliations

When a consequential tool raises in a checkpointed run, its outcome is
unknown, and the run pauses until an operator checks the external system and
reconciles the action in agentbarrier. `agent.resume(config)` then retries the
call: a committed result is replayed, a not-committed action runs exactly
once, and an unreconciled action pauses the run again. See
[Failures and reconciliation](../concepts/policies.md#failures-and-reconciliation).

### Tool-call review

LangChain's `HumanInTheLoopMiddleware` pauses before selected tools run, so a
person can approve, edit, or reject each call. It is standard middleware, so
it applies to every tool-capable agent in every topology:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

from agloom import ExecutionPaused


@tool
def restock(sku: str, units: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"ordered {units} units of {sku}"


supervised = create_agent(
    model=model,
    tools=[restock],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"restock": True})],
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=authorize,
)
config = {"configurable": {"thread_id": "case-19"}}
try:
    supervised.invoke("Restock 40 units of SKU-778.", config)
except ExecutionPaused as paused:
    for action in paused.interrupts[0].value["action_requests"]:
        print(action["name"], action["args"])
    answer = supervised.resume(config, {"decisions": [{"type": "approve"}]})
    print(answer.content)
```

Resume with one decision per action request, in order:
`{"type": "approve"}`, `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`,
`{"type": "reject", "message": "..."}`, or `{"type": "respond", "message": "..."}`.
Limit the choices per tool with
`interrupt_on={"restock": {"allowed_decisions": ["approve", "reject"]}}`.
Tool-call review is a conversational control. For actions that must be
authorized against rules, bound to their exact arguments, and executed exactly
once with an audit trail, use [policy approvals](../concepts/policies.md#approvals).

### Breakpoints

Breakpoints are debugging stops: they pause a topology before or after a node
so that you can inspect the run, then continue it. Node names are topology
specific, so breakpoints require one explicit `pattern`
([`ConfigurationError`][agloom.ConfigurationError] otherwise), and recursive
children never stop at them. The interrupt payload lists the configured
breakpoints: `{"type": BREAKPOINT_INTERRUPT, "interrupt_before": [...],
"interrupt_after": [...]}`. To have a person decide on tool calls before they
run, use tool-call review or policy approvals instead.

For example, stop a ReAct run before its `tools` node, inspect it, and continue:

```python
from agloom import BREAKPOINT_INTERRUPT, ExecutionPaused

debugged = create_agent(
    model=model,
    tools=[order_status],
    pattern="react",
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=authorize,
    interrupt_before=["tools"],
)
config = {"configurable": {"thread_id": "case-18"}}
try:
    debugged.invoke("Where is order A-1001?", config)
except ExecutionPaused as paused:
    assert paused.interrupts[0].value["type"] == BREAKPOINT_INTERRUPT
    answer = debugged.resume(config)  # run the tool call and finish
    print(answer.content)
```

See [Topologies](../concepts/topologies.md) for every topology's node names.

??? example "examples/20_hitl.py: tool-call review and breakpoints"

    ```python
    --8<-- "examples/20_hitl.py"
    ```

### Loop pauses

With `LoopPolicy(allow_human_intervention=True)`, the loop can pause for a
person, for example when progress stagnates or an evaluator asks for help. The
resume value becomes standing human feedback: it applies to this and every
later iteration (the latest 10 notes, up to 2,000 characters each):

```python
answer = agent.resume(config, "Focus on orders shipped after 1 October.")
```

You can also pass `{"feedback": ["first note", "second note"]}`.

## Resume semantics

`agent.resume(config, value=True)` and `await agent.aresume(...)` continue a
thread:

- every pending interrupt of the thread receives `value`;
- a thread stopped at a breakpoint, or interrupted by a crash or a
  cancellation, continues from its last checkpoint;
- completed steps, including completed loop iterations and executed
  consequential tools, are not repeated;
- the run keeps the plan it paused in, with the same capabilities and
  topology at every recursion depth, instead of analyzing the task again;
- a thread with nothing to resume raises [`ConfigurationError`][agloom.ConfigurationError].

`agent.invoke(None, config)` and `agent.invoke_loop(None, config)` (and their
async twins) also continue a checkpointed thread.

Inspect a thread at any time with `agent.get_state(config)` (or
`aget_state`); it returns the LangGraph `StateSnapshot`, including pending
interrupts.

## Agents inside another graph

An agent can run as a step of your own LangGraph graph or as a Deep Agents
sub-agent. Pass the node's `config` to the agent, and give the agent the
enclosing graph's checkpointer, so the agent runs in the enclosing thread: a
pause then reaches your graph as a LangGraph interrupt, and resuming your graph
resumes the agent.

```python
from agentbarrier import Decision
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

saver = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[lookup_order, refund_order],
    policy=policy,  # the policy from "Policies and approvals"
    checkpointer=saver,
    checkpoint_authorizer=authorize,
)


class Ticket(TypedDict, total=False):
    request: str
    answer: str


def support(state: Ticket, config: RunnableConfig) -> dict[str, str]:
    return {"answer": agent.invoke(state["request"], config).text}


graph = (
    StateGraph(Ticket)
    .add_node("support", support)
    .add_edge(START, "support")
    .add_edge("support", END)
    .compile(checkpointer=saver)
)
config = {"configurable": {"thread_id": "case-18"}}
request = "Refund order A-1042 in full. Use refund request id R-88."
paused = graph.invoke({"request": request}, config)
approval = paused["__interrupt__"][0].value
policy.store.decide(approval["action_id"], Decision.APPROVE, decided_by="alice")
answer = graph.invoke(Command(resume=True), config)["answer"]
```

An agent invoked with its own `thread_id` instead runs as a separate thread:
its pauses raise [`ExecutionPaused`][agloom.ExecutionPaused] inside your node,
and you resume it with `agent.resume`.

!!! note "Resuming nested graphs"

    LangGraph currently re-runs the work that already finished in the
    interrupted step of a nested graph when it resumes
    ([langchain-ai/langgraph#9106](https://github.com/langchain-ai/langgraph/issues/9106)).
    The agent's plans, finished topology results, and agentbarrier's
    exactly-once actions are not affected, but in an agent paused inside
    another graph, the loop evaluations and refreshes of earlier iterations
    run again. Run a multi-iteration loop that can pause as the top-level
    graph.

## Recovering after a crash

When a process dies mid-run, the checkpoint keeps every completed step. Call
`agent.resume(config)` with the same `thread_id` from any process that shares
the checkpointer: the run continues without repeating completed iterations or
tool calls. Consequential tools are protected twice: by the checkpoint, and by
agentbarrier's exactly-once claim, so a call that was running when the process
died is never repeated silently.

In production, run agents in a durable job queue and re-enqueue interrupted
runs; see [Production deployment](production.md#background-execution).

??? example "examples/21_checkpoint_resume.py"

    ```python
    --8<-- "examples/21_checkpoint_resume.py"
    ```
