# Architecture version

**Agloom architecture: 2.0 (frozen before implementation)**  
**Status:** Baseline for this implementation. Changes require an explicit decision
and updates to affected architecture documents and tests.

This specification defines eight actual execution topologies—ReAct, Supervisor,
Pipeline, Planner, Reflection, Swarm, Blackboard, and Hybrid—and the separate
DIRECT execution mode. DIRECT is not a topology. Recursion is a bounded,
orthogonal execution capability, not a topology.

The implementation target is Python 3.12, LangChain's model/tool interfaces,
and LangGraph's graph runtime. LangSmith is not required. The core package
must not depend on provider-specific model abstractions or hosted services.

## Frozen decisions

1. `create_agent()` is the public construction API. A one-pattern constraint
   bypasses task analysis; multiple allowed patterns invoke constrained
   analysis; no constraint invokes automatic analysis.
2. `TaskAnalysis` is structured data. `StrategyEngine` turns it into an
   `ExecutionDecision`. Its separate `ExecutionBlueprint` contains bounded,
   task-specific execution recommendations but no decision; only a validated
   `ArchitectureSpec` reaches a compiler.
3. DIRECT invokes the configured model without constructing or compiling a
   topology graph. Tool availability alone does not imply tool use; tasks that
   require tools select ReAct and use LangChain's tool abstractions.
4. Topologies share a registry, state contract, lifecycle, event surface,
   bounded worker policy, and runtime. A topology contributes execution
   semantics, not a second runtime.
5. LangGraph is the graph execution and checkpoint integration foundation.
   Checkpointing is opt-in and uses caller-provided LangGraph checkpointers.
6. Recursion is disabled by default, and when enabled is bounded by explicit
   depth, task, worker, and elapsed-time policy.
7. Named ecosystem packages are integrated through thin adapters against their
   real public APIs. All seven named packages are required runtime dependencies;
   their capabilities can be explicitly configured or injected. An abstract
   agent-local capability catalog combines LangChain tools, application
   services, and MCP capabilities. Routing applies policy plus optional
   LangChain embedding, retrieval, and reranking contracts before capability
   metadata reaches analysis and execution. Current PyPI metadata shows several
   packages require Python <3.13, so Python 3.12 is the compatible target.
8. Human approval is an explicit interrupt/resume boundary. Streaming uses
   LangGraph/LangChain event or stream interfaces, with no custom model protocol.
9. Failures are typed and surfaced. Automatic mode may select DIRECT when
   analysis confidently finds no orchestration need; it does not silently
   downgrade failed topology execution.
10. Untrusted message inputs cannot supply system, developer, or tool roles;
    trusted instructions are configured separately with `system_prompt`.
11. Every checkpoint operation requires a caller-provided authorizer that
    binds the requested `thread_id` to the current authenticated principal.

## Change control

The implementation phase may refine internal names and module boundaries, but
must preserve these decisions. Any material behavioral change first updates this
specification, its dependent documents, and acceptance tests.
