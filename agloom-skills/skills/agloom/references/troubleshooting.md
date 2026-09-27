# Debugging and testing Agloom

Reproduce the failure before changing configuration or application code.
Capture the exact `create_agent()` arguments, input, sync/async method, selected
mode or topology, capability inventory, lifecycle state, and chained exception.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

## Investigation sequence

1. **Confirm the boundary.** Determine whether the failure belongs to Agloom
   selection/runtime or to the model provider, tool, MCP server, database,
   checkpointer, observer, or application.
2. **Inspect explicit intent.** One pattern bypasses analysis; a list constrains
   it; Hybrid requires `composition`.
3. **Inspect available capabilities.** Confirm required tools/services were
   registered, policy permits them, and retriever/reranker documents carry the
   expected capability name.
4. **Inspect the real call mode.** Use `ainvoke`/`astream` inside active event
   loops. Do not hide an async boundary behind synchronous application code.
5. **Inspect topology requirements.** Supervisor, Pipeline, Hybrid, and other
   topologies may require explicit or analyzed workers/stages/composition.
6. **Inspect recursion bounds.** Confirm recursion is enabled and depth, task,
   worker, topology, and elapsed-time limits permit the child.
7. **Inspect checkpoint configuration.** Confirm a checkpointer, authorizer,
   stable thread ID, and graph-backed topology are present.
8. **Inspect MCP lifecycle.** Confirm client/server names, discovery options,
   remote server health, and async initialization.
9. **Inspect integration option keys.** They are validated by installed
   upstream package constructors.
10. **Inspect observability separately.** Flush before handoff, close at
    shutdown, and surface observer failures.
11. **Compare with a runnable example.** Reproduce the smallest matching
    [example](https://github.com/smuniharish/agloom/tree/master/examples).

## Common diagnoses

| Symptom | Verify first |
| --- | --- |
| A greeting uses a topology | Pattern constraints, task wording, and automatic-analysis output |
| A tool-required task fails selection | Tool registration, description, policy, and capability inventory |
| Explicit Planner still analyzes | `pattern` is one string, not a one-item sequence transformed elsewhere |
| Pattern list runs multiple topologies | The application incorrectly interpreted an allow-list as composition |
| Hybrid construction fails | Two to four non-Hybrid children are supplied |
| Child task raises a recursion error | `recursion=True` and remaining policy budget |
| Sync MCP call fails in an event loop | Replace it with the async agent/capability API |
| Resume or state read is rejected | Authorizer decision and matching `configurable.thread_id` |
| Structured output is absent | Schema configured and `agent.structured_output` used |
| Metrics or traces are missing | Observer initialization, callback propagation, endpoint, flush, and close |
| Custom router construction fails | Do not combine a complete router with default-router component options |

## Focused verification

Assert an observable property rather than generated wording:

- exact selected mode/topology;
- exact tool call and arguments;
- selected capability names;
- explicit unavailable-capability failure;
- bounded recursion error;
- pause/resume state;
- structured model value;
- emitted event or metric;
- absence of topology compilation on DIRECT.

For runtime changes, run the formatter, linter, type checker, focused tests,
full tests when shared behavior changes, and strict documentation build. For
skill-only changes, follow
[`../../../validation/README.md`](../../../validation/README.md).
