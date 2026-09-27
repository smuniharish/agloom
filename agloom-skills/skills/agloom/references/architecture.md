# Agloom architecture for coding agents

Use this reference to determine what Agloom owns before changing an
application or proposing an extension.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

## Ownership boundary

Agloom owns:

- `create_agent()` configuration and validation;
- capability-aware task classification;
- DIRECT versus topology selection;
- eight topology execution semantics;
- LangGraph compilation and runnable adaptation;
- per-agent lifecycle, events, cancellation, streaming, and recursion budgets;
- capability cataloging, policy, selection, and named execution;
- integration adapters for the seven supported ecosystem packages.

It delegates infrastructure to its owner:

| Responsibility | Owner |
| --- | --- |
| Model provider, credentials, and inference behavior | LangChain provider and application |
| Tool business logic and application services | Application |
| Graph execution and checkpoint semantics | LangGraph |
| Checkpoint persistence and thread authorization | Application |
| MCP transport and remote server lifecycle | Application and MCP ecosystem |
| Feedback, behavior, context, refresh, xAI, structured output | Named ecosystem packages |
| Database, vector store, embeddings service, and reranker service | Application |
| Authentication, tenancy, deployment, and infrastructure scaling | Application |

Agloom scales **orchestration complexity** from DIRECT to bounded topologies.
It does not autoscale compute infrastructure.

## Execution resolution

```text
create_agent configuration
          |
          v
    pattern supplied?
      /          \
 one value     none or list
    |               |
 explicit       task analysis
 topology           |
    |          constrained choice
    +-------+-------+
            |
            v
       DIRECT or one topology
            |
            v
       isolated runtime
```

- No pattern: DIRECT or any topology may be selected.
- One pattern: use that topology without automatic selection.
- Several patterns: select one candidate from the allow-list.
- Hybrid: an explicit topology with two to four non-Hybrid children.

## State and isolation

Each `create_agent()` call owns its runtime configuration and observer pipeline.
Each invocation receives fresh lifecycle, cancellation, and recursion state.
Caller-provided models, tools, checkpointers, and capability implementations are
references and remain the caller's concurrency responsibility.

## Recursion and Hybrid

Recursion is bounded child-task resolution and is orthogonal to topology.
Hybrid is explicit topology composition. Neither term can replace the other,
and neither creates an unbounded autonomous loop.
