# Architecture compiler

The compiler accepts only a validated `ArchitectureSpec`, never raw analyzer
output. The specification identifies DIRECT or a registered topology, topology
configuration, input/output contracts, capability handles, recursion settings,
and graph/runtime options.

Compilation validates topology and composition constraints. DIRECT produces a
direct runner and no graph. A topology builder produces a LangGraph
`StateGraph`; the common compiler applies the state schema and compiles it with
the configured checkpointer. Validation errors are `CompilationError` or
`TopologyError` with preserved causes. A topology cannot be silently substituted.

New topology extension: implement the topology contract, register the builder,
validate its config, add tests/docs; the public factory and runtime remain
unchanged.
