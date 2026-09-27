# Examples and verified results

Agloom includes focused scripts and five complete FastAPI/React applications.
The scripts use the real OpenAI-compatible endpoint configured through
`EXPLABS_API_KEY`, `EXPLABS_BASE_URL`, and `EXPLABS_MODEL`. Credentials stay in
the environment; no example embeds a key or silently substitutes a fake model.

## DIRECT avoids unnecessary tool calls

The DIRECT example deliberately supplies a tool that the request does not
need:

```python
tool_calls = 0

@tool
def lookup_exchange_rate(currency: str) -> str:
    """Return a sample exchange rate for a currency."""
    global tool_calls
    tool_calls += 1
    return f"1 {currency} = 1.08 USD"


agent = create_agent(model=real_model(), tools=[lookup_exchange_rate])
result = agent.invoke("Compute 6 multiplied by 7 and include the number 42.")

assert tool_calls == 0
assert "42" in result.content
```

**Verified result:** the response contains `42`, the exchange-rate tool is
never called, and no topology graph is required.

## Automatic selection uses a relevant tool

```python
tool_calls = []

@tool
def average_speed(distance_km: float, duration_hours: float) -> float:
    """Calculate average speed in kilometers per hour."""
    tool_calls.append((distance_km, duration_hours))
    return distance_km / duration_hours


agent = create_agent(model=real_model(), tools=[average_speed])
result = agent.invoke(
    "Use average_speed for 120 kilometers in 2 hours. Return km/h."
)

assert tool_calls == [(120.0, 2.0)]
assert "60" in result.content
assert "km/h" in result.content
```

**Verified result:** capability-aware selection chooses tool-capable execution,
calls `average_speed(120.0, 2.0)` exactly once, and returns `60 km/h`.

## Explicit ReAct

```python
agent = create_agent(
    model=real_model(),
    tools=[multiply],
    pattern="react",
)
result = agent.invoke("Use multiply to calculate 17 times 23.")

assert tool_calls == [(17, 23)]
assert "391" in result.content
```

**Verified result:** an explicit single topology bypasses automatic selection,
the tool receives the exact operands, and the answer contains `391`.

## Complete configuration

`examples/24_complete_configuration.py` constructs compatible agents that
collectively pass every public `create_agent()` parameter. It verifies:

- local tools, application capabilities, and MCP tools/resources/prompts;
- policy filtering, embeddings, retrieval, and reranking;
- FeedbackManager, BehaviorWeave, ContextSage, and RefreshEngine;
- Prometheus/Langfuse observers and langgraph-xai executions;
- xstructured output, checkpoint resume, Hybrid composition, and custom
  strategy/compiler/router injection.

The script compares its option coverage with
`inspect.signature(create_agent)`. Adding a public parameter without adding a
working example therefore fails the example:

```python
parameters = set(inspect.signature(create_agent).parameters)
assert covered == parameters
print(f"create_agent parameter coverage: {len(parameters)}/{len(parameters)}")
```

Representative verified results include:

```text
ReAct: response contains 96
MCP tool/resource/prompt: verified
Feedback lifecycle: resolved
Behavior policy: pause
Refresh: one modified document
Structured output: result=42 verified=True
Checkpoint resume: response contains 24
Hybrid: response contains 25
create_agent parameter coverage: complete
```

Response prose can vary by model; the examples assert semantic values, exact
tool arguments, lifecycle states, and structured objects rather than brittle
word-for-word output.

## Focused scripts

| Area | Runnable scripts |
|---|---|
| Selection and topology behavior | `01_direct.py` through `10_hybrid.py` |
| Recursion | `11_recursion.py`, `12_no_recursion.py` |
| Capability packages | `13_mcp.py` through `19_structured_output.py` |
| Runtime controls | `20_hitl.py`, `21_checkpoint_resume.py`, `22_streaming.py` |
| Observability | `23_observability.py` |
| Every public construction option | `24_complete_configuration.py` |

Run a script from the repository root:

```console
uv run python examples/04_react_tools.py
```

## Complete applications

| Application | What it demonstrates |
|---|---|
| Enterprise Knowledge Assistant | Retrieval, citations, source ingestion, feedback, and observability |
| QA Customer Chatbot | Customer support tools, operational data, guardrails, and feedback |
| RCA Generator | Incident evidence, structured root-cause analysis, and approval workflows |
| Web Compliance Auditor | Playwright/Filesystem/Everything MCP, Ollama embeddings, pgvector, Grafana, and Langfuse |
| Engineering Change Investigator | Engineering records, live browser evidence, durable state, pgvector, Grafana, and Langfuse |

Each application is an independent uv project with a locked backend,
production container build, React frontend, seeded data, and tests. Follow the
application README under `examples/<application>/`.

## Current verification

The documentation update was checked against the executable repository:

| Verification | Result |
|---|---:|
| Agloom framework tests | 214 passed |
| Five isolated application suites | 66 passed |
| Example backend production images | 5 built and import-smoked |
| Example Compose configurations | 5 valid |
| Ruff, Black, Pyrefly, and strict MkDocs | Passed |

These counts describe the current repository validation, not a promise that
provider latency or generated prose will be identical across environments.
