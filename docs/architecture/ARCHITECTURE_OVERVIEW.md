# Architecture overview

Agloom is a library harness around LangChain and LangGraph. It chooses whether a
task can run directly or requires one of eight execution topologies, compiles
the selected topology to LangGraph, and runs it through an isolated runtime.

<picture class="agloom-diagram">
<source media="(max-width: 600px)" srcset="../../assets/diagrams/architecture-overview-mobile.png">
<img src="../../assets/diagrams/architecture-overview.png" alt="Agloom request flow from create_agent through selection and execution">
</picture>

The public contract is intentionally simple: configuration determines the
allowed execution modes, capability-aware analysis is used only when selection
is automatic, and every invocation runs in isolated state. Application-owned
models, tools, checkpointers, and capability providers remain under application
control.
