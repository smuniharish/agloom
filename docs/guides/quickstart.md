# Quickstart

Construct an agent from any supported LangChain chat model. This repository's
examples use the real OpenAI-compatible endpoint configured by environment:

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
    model=os.getenv("EXPLABS_MODEL", "gpt-5.6-luna"),
    base_url=os.getenv(
        "EXPLABS_BASE_URL",
        "https://api.experientiallabs.ai/v1",
    ),
    api_key=os.environ["EXPLABS_API_KEY"],
)
agent = create_agent(model=model, tools=[multiply])
answer = agent.invoke("Use multiply to calculate 17 times 23.")
print(answer.content)
```

Keep credentials in the environment. Copy `.env.example` only as a variable
template; never place a real key in source control.

The response wording is model-dependent, but the verified semantic result is
`391`, and the ReAct execution calls `multiply(17, 23)` exactly once. The
runnable example asserts both conditions in `examples/04_react_tools.py`.

## Selection options

```python
# The capability-aware LLM analyzer chooses DIRECT or a topology.
automatic = create_agent(model=model, tools=[multiply])

# A single explicit topology bypasses analysis.
planner = create_agent(model=model, tools=[multiply], pattern="planner")

# An allow-list selects one candidate; this does not compose them.
constrained = create_agent(
    model=model,
    tools=[multiply],
    pattern=["planner", "supervisor"],
)
```

Automatic and allow-list modes perform an analysis model call before execution.
The analyzer sees configured tool names and descriptions, named capabilities,
workers, and pipeline stages. It rejects tasks that require unavailable
capabilities instead of silently routing them to DIRECT. Explicit developer
workers, stages, prompts, and composition always take precedence.

For composition, use the Hybrid topology and provide its child topologies
explicitly. See [choosing an execution mode](execution.md#hybrid-composition).

Every agent is explainable by default through an isolated `langgraph-xai`
runtime. Access it as `agent.xai`, or pass `xai_options` to override the
upstream runtime defaults.
