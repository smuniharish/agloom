# Runtime, checkpointing, and approval

Each `Agent` is isolated and can be invoked, streamed, paused, resumed, and
stopped independently. Applications may use a `RuntimeController` to request
cooperative cancellation across a known set of agents.

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

See the [production runtime guide](../guides/production.md) and
[HITL example](../examples/overview.md).
