# Runtime, checkpointing, and approval

Each `Agent` owns its event sink, active invocation cancellation signals,
compiled graph cache, and recursion sessions. Applications may use a
`RuntimeController` to coordinate cooperative stop across explicitly
registered agents.

LangGraph checkpointing is opt-in and requires a caller-provided checkpointer
and a stable `configurable.thread_id`. Applications must also provide a
`checkpoint_authorizer` that verifies the thread ID against their current
authenticated principal. Agloom invokes it for graph runs, state reads, and
resumes; a thread ID is not itself an authorization credential. Interrupts can pause graph execution;
resume with a LangGraph `Command`:

```python
from langgraph.types import Command

paused = agent.invoke(task, config=run_config)
resumed = agent.invoke(Command(resume=True), config=run_config)
```

See the [harness runtime](../architecture/HARNESS_RUNTIME.md), [lifecycle
model](../architecture/LIFECYCLE_MODEL.md), and [HITL example](../examples/overview.md).
