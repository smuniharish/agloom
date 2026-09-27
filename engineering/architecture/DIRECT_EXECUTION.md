# DIRECT execution

DIRECT invokes the configured LangChain chat model with the request and
supported messages. Merely making tools available does not make a task require
tool use: automatic analysis keeps straightforward requests on DIRECT. Tasks
that need tools select ReAct, which uses LangChain's tool abstractions.

<picture class="agloom-diagram">
<source media="(max-width: 600px)" srcset="../../assets/diagrams/direct-execution-mobile.png">
<img src="../../assets/diagrams/direct-execution.png" alt="DIRECT execution sends a task directly to the model and returns its response">
</picture>

Explicit topology configuration never resolves to DIRECT. Automatic analysis
may select DIRECT for a straightforward task. Direct invocation still emits
runtime events and honors cooperative cancellation and timeouts when supported.
It does not construct a topology graph, which keeps the simple path lightweight.
