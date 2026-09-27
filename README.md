# Agloom

## Production-grade Python agent harness that automatically classifies tasks, optimizes execution, and scales orchestration across eight specialized topologies.

Agloom is a production-grade Python agent harness that automatically classifies
tasks, optimizes execution, and scales orchestration from lightweight DIRECT
execution to eight specialized LangGraph topologies for efficient, reliable
agent workloads.

[Documentation](https://agloom.readthedocs.io/en/latest/) |
[Why Agloom?](https://agloom.readthedocs.io/en/latest/guides/why-agloom/) |
[Quickstart](https://agloom.readthedocs.io/en/latest/guides/quickstart/) |
[Examples](https://agloom.readthedocs.io/en/latest/examples/overview/) |
[API reference](https://agloom.readthedocs.io/en/latest/api/reference/) |
[GitHub](https://github.com/smuniharish/agloom)

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangChain 1.x](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![LangGraph 1.2](https://img.shields.io/badge/LangGraph-1.2-1C3C3C)
![Apache License 2.0](https://img.shields.io/badge/License-Apache--2.0-blue)

---

Building one LLM call is easy. Building an agent that stays predictable when a
task needs tools, planning, multiple workers, long context, human approval,
remote MCP capabilities, or durable execution is not.

Agloom gives applications one construction API:

```python
from agloom import create_agent

agent = create_agent(model=model, tools=tools)
```

For each request, Agloom can keep simple work on lightweight **DIRECT
execution** or select one of eight bounded execution topologies when
orchestration is useful. Explicit developer configuration always wins.

> **Agloom provides eight execution topologies plus a direct execution mode.**
> DIRECT is not a topology.

## Why teams use Agloom

| Agent engineering problem | Agloom's answer |
|---|---|
| Every request pays the cost of a planner or graph | Straightforward requests use DIRECT execution |
| A single workflow becomes a maze of conditional branches | Eight topologies provide clear, reusable execution semantics |
| Automatic routing plans work the agent cannot perform | Selection uses the actual tool and capability inventory |
| Local tools, application services, and MCP become separate systems | One policy-aware capability catalog routes all three |
| Recursive and multi-agent work grows without bounds | Explicit depth, task, worker, topology, and time limits |
| Pause, resume, streaming, and cancellation are added ad hoc | LangGraph-native runtime controls are available through the agent |
| Model, tool, and graph activity is difficult to connect | Structured events, Prometheus, Grafana, Langfuse, and langgraph-xai |
| Typed responses require custom parsing and repair | xstructured validates and repairs structured output |

## Quickstart

Install Agloom and your model provider:

```bash
uv add agloom langchain-openai
```

Or with pip:

```bash
python -m pip install agloom langchain-openai
```

Create an agent with a real LangChain model and tool:

```python
import os

from agloom import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right


model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    api_key=os.environ["OPENAI_API_KEY"],
)

agent = create_agent(model=model, tools=[multiply])
response = agent.invoke("Use multiply to calculate 17 times 23.")

print(response.content)
```

Representative result:

```text
17 × 23 = 391.
```

The executable repository example also asserts that `multiply(17, 23)` is
called exactly once and that the semantic answer contains `391`.

## Automatic when useful, explicit when required

```python
# Agloom may select DIRECT or any of the eight topologies.
automatic = create_agent(model=model, tools=tools)

# One explicit topology bypasses automatic selection.
planner = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
)

# An allow-list constrains selection to one of these candidates.
constrained = create_agent(
    model=model,
    tools=tools,
    pattern=["planner", "supervisor"],
)
```

An allow-list is not a composition. Use the Hybrid topology explicitly when a
task should pass through multiple child topologies:

```python
hybrid = create_agent(
    model=model,
    tools=tools,
    pattern="hybrid",
    composition=["react", "reflection"],
)
```

## Eight topologies, one runtime

| Topology | Best suited for |
|---|---|
| **ReAct** | Reasoning with tool calls and observations |
| **Supervisor** | Routing work across a finite set of specialist workers |
| **Pipeline** | Ordered stages with explicit result propagation |
| **Planner** | Decomposing a task into bounded subtasks and aggregating results |
| **Reflection** | Drafting, evaluating, and revising an answer |
| **Swarm** | Bounded handoffs between peer workers |
| **Blackboard** | Collaboration through a versioned shared workspace |
| **Hybrid** | Explicit composition of two to four non-Hybrid topologies |

Every topology shares the same lifecycle, events, capability routing,
checkpoint integration, streaming surface, and recursion policy.

## One capability catalog

Agloom treats capabilities as independent from topology:

```text
Local LangChain tools ─┐
Application services ──┼─> policy ─> retrieval ─> reranking ─> selection
MCP capabilities ──────┘
```

```python
agent = create_agent(
    model=model,
    tools=local_tools,
    capabilities={"document_store": document_store},
    mcp_client=mcp_client,
    mcp_server_name="operations",
    capability_policy=policy,
    capability_embeddings=embeddings,
    capability_retriever=retriever,
    capability_reranker=reranker,
    max_selected_capabilities=12,
)

selection = await agent.aroute_capabilities(
    "Find the deployment guide and current service status."
)
print(selection.names)
```

Policy filtering runs first. Optional semantic retrieval and reranking then
reduce the catalog to the capabilities relevant to the request. If a task
requires a capability that is unavailable, Agloom fails explicitly instead of
asking the model to pretend.

## Production controls without a new abstraction

```python
from agloom import RecursionPolicy

agent = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(
        max_depth=2,
        max_subtasks=8,
        max_workers=4,
        max_execution_time=120.0,
    ),
    observers=[prometheus, langfuse],
    structured_output_schema=IncidentReport,
)
```

The same runnable API can provide:

- synchronous and asynchronous invocation;
- streaming;
- cooperative cancellation;
- bounded recursive child execution;
- LangGraph checkpointing, interrupts, and human approval;
- typed structured output;
- agent-local explainability through `agent.xai`;
- lifecycle, model, retriever, and tool telemetry.

Agloom does not require LangSmith, a database, containers, or hosted
infrastructure. Applications opt into the infrastructure their capabilities
need.

## Integrated ecosystem

Agloom installs and integrates these capability packages:

| Capability | Package |
|---|---|
| Feedback lifecycle | `feedback-manager` |
| Behavioral policies | `behaviorweave` |
| Context management | `contextsage` |
| MCP discovery and routing | `mcp-capability-router` |
| Source refresh | `refresh-engine` |
| Explainability | `langgraph-xai` |
| Structured output | `xstructured` |

Their supported configuration is exposed through `create_agent()` rather than
through a parallel service layer. See the [capability
guide](https://agloom.readthedocs.io/en/latest/concepts/capabilities/).

## Real examples, not only toy prompts

The repository includes focused scripts for every topology and integration,
plus five complete FastAPI/React applications:

| Application | Demonstrates |
|---|---|
| Enterprise Knowledge Assistant | Retrieval, citations, ingestion, feedback, and observability |
| QA Customer Chatbot | Support tools, operational data, behavioral guardrails, and feedback |
| RCA Generator | Incident evidence, structured analysis, and approval workflows |
| Web Compliance Auditor | Real Playwright, Filesystem, and Everything MCP servers; Ollama embeddings; pgvector |
| Engineering Change Investigator | MCP evidence, engineering records, PostgreSQL/pgvector, durable integration state |

`examples/24_complete_configuration.py` constructs compatible agents that
collectively exercise **every public
`create_agent()` parameter** and fails if a newly added parameter is not
demonstrated.

See [examples and verified
results](https://agloom.readthedocs.io/en/latest/examples/overview/) for executable
snippets, semantic assertions, and expected outcomes.

## Documentation

- [Why Agloom?](https://agloom.readthedocs.io/en/latest/guides/why-agloom/)
- [Installation](https://agloom.readthedocs.io/en/latest/guides/installation/)
- [Quickstart](https://agloom.readthedocs.io/en/latest/guides/quickstart/)
- [Execution topologies](https://agloom.readthedocs.io/en/latest/concepts/topologies/)
- [Capabilities](https://agloom.readthedocs.io/en/latest/concepts/capabilities/)
- [Runtime and human approval](https://agloom.readthedocs.io/en/latest/concepts/runtime/)
- [Observability](https://agloom.readthedocs.io/en/latest/concepts/observability/)
- [Public API guide](https://agloom.readthedocs.io/en/latest/architecture/PUBLIC_API/)
- [Hosted documentation](https://agloom.readthedocs.io/en/latest/)
- [GitHub repository](https://github.com/smuniharish/agloom)

## Compatibility

Agloom targets **Python 3.12** and is designed as a production-grade,
embeddable framework. It does not require LangSmith, containers, a database, or
a hosted control plane.

## Author

**S Muni Harish**  
[samamuniharish@gmail.com](mailto:samamuniharish@gmail.com)

Agloom is licensed under the Apache License 2.0.
