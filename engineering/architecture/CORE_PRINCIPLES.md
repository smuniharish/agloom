# Core principles

1. Keep simple tasks simple: DIRECT must not analyze via an unnecessary model,
   instantiate a topology, or compile a graph.
2. Preserve explicit intent: a single topology constraint is deterministic;
   a multi-candidate constraint never selects outside its allow-list.
3. Separate observation, choice, specification, compilation, and execution.
4. Reuse LangChain messages, models, tools, and runnable contracts; reuse
   LangGraph state graphs, interrupts, streaming, and checkpointers.
5. Keep topologies composable and independently testable through one runtime.
6. Bound recursive work and concurrency before launching child tasks.
7. Keep capabilities topology-independent, optional, and explicit.
8. Isolate per-agent state and policies; avoid implicit process-wide mutable
   state.
9. Make failures observable and preserve their cause and execution context.
10. Favor useful behavior and verified integrations over placeholder breadth.

Acceptance checks are encoded in tests and the architecture terminology audit.
