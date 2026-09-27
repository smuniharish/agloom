# Verified integration workflows

Import public symbols from `agloom`. Do not use private submodules for normal
application integration.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

## Install

Agloom supports Python 3.12:

```bash
uv add agloom langchain-openai
```

Keep provider credentials outside source control.

## Automatic execution with a tool

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
result = agent.invoke("Use multiply to calculate 17 times 23.")
print(result.content)
```

Automatic selection can keep simple work on DIRECT and choose tool-capable
execution when the request requires `multiply`.

## Explicit and constrained selection

```python
planner = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
)

constrained = create_agent(
    model=model,
    tools=tools,
    pattern=["planner", "supervisor"],
)

hybrid = create_agent(
    model=model,
    tools=tools,
    pattern="hybrid",
    composition=["react", "reflection"],
)
```

The constrained agent selects one candidate. Only the Hybrid agent composes
topologies.

## Bounded recursion

```python
from agloom import RecursionPolicy, create_agent

agent = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(
        max_depth=2,
        max_workers=4,
        max_subtasks=8,
        max_execution_time=120.0,
    ),
)
```

Recursion is disabled by default. Every child shares the invocation's bounded
budget and cancellation path.

## Checkpoint and resume

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model=model,
    tools=[],
    pattern="pipeline",
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=lambda thread_id: thread_id == "review-42",
    interrupt_before=["stage_0_analyze"],
)
config = {"configurable": {"thread_id": "review-42"}}

paused = agent.invoke("Prepare the release review.", config=config)
resumed = agent.invoke(Command(resume=True), config=config)
```

Production applications should use durable persistence and authorize thread
IDs against the authenticated principal.

## Supported example routes

| Need | Authoritative example |
| --- | --- |
| DIRECT without unnecessary tools | [`01_direct.py`](https://github.com/smuniharish/agloom/blob/master/examples/01_direct.py) |
| Automatic selection | [`02_automatic.py`](https://github.com/smuniharish/agloom/blob/master/examples/02_automatic.py) |
| ReAct tool execution | [`04_react_tools.py`](https://github.com/smuniharish/agloom/blob/master/examples/04_react_tools.py) |
| Hybrid composition | [`10_hybrid.py`](https://github.com/smuniharish/agloom/blob/master/examples/10_hybrid.py) |
| Bounded recursion | [`11_recursion.py`](https://github.com/smuniharish/agloom/blob/master/examples/11_recursion.py) |
| MCP capabilities | [`13_mcp.py`](https://github.com/smuniharish/agloom/blob/master/examples/13_mcp.py) |
| Structured output | [`19_structured_output.py`](https://github.com/smuniharish/agloom/blob/master/examples/19_structured_output.py) |
| Human approval | [`20_hitl.py`](https://github.com/smuniharish/agloom/blob/master/examples/20_hitl.py) |
| Streaming | [`22_streaming.py`](https://github.com/smuniharish/agloom/blob/master/examples/22_streaming.py) |
| Observability | [`23_observability.py`](https://github.com/smuniharish/agloom/blob/master/examples/23_observability.py) |
| Every public construction option | [`24_complete_configuration.py`](https://github.com/smuniharish/agloom/blob/master/examples/24_complete_configuration.py) |

The five complete FastAPI/React projects under
[examples](https://github.com/smuniharish/agloom/tree/master/examples) show
application-level persistence, MCP, embeddings, retrieval, observability, and
governance.
