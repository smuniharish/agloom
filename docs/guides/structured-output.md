# Structured output

`agent.with_structured_output(schema)` returns a runnable that produces a
validated object together with the prose answer, whatever execution mode or
topology the request runs in. It is powered by
[xstructured](https://pypi.org/project/xstructured/).

```python
from pydantic import BaseModel


class OrderReport(BaseModel):
    order_id: str
    status: str
    delivery_date: str | None = None


structured = agent.with_structured_output(OrderReport)
result = structured.invoke("Report the status of order A-1002.")
print(result.structured)
```

```text
order_id='A-1002' status='in transit' delivery_date='8 October'
```

## How it works

1. xstructured's schema instructions are appended to the latest user message,
   so the runnable accepts the same inputs as the agent: text, a list of
   messages, or a mapping with `messages`.
2. The agent runs as usual: analysis, DIRECT or a topology, tools, policies,
   and Loop Engineering.
3. xstructured extracts the tagged JSON from the answer and validates it with
   Pydantic.

Because validation happens on the final answer, it works for every topology,
including multi-agent ones, and does not require provider-native structured
output. The structured runnable is itself a LangChain runnable: use `ainvoke`
in async code and compose it in LCEL chains.

```python
from langchain_core.messages import HumanMessage

result = await structured.ainvoke([HumanMessage("Report the status of order A-1001.")])
print(result.structured)
```

## The result

`invoke` returns an `XStructuredResult`:

| Attribute | Contains |
|---|---|
| `structured` | The validated object |
| `content` | The prose of the answer, without the structured block |
| `raw` | The agent's original answer message |
| `recovered`, `repaired`, `repair_attempts` | Whether conservative recovery or model repair was needed |
| `metadata` | Schema fingerprint, parse duration, and message metadata |

## Options

Keyword arguments are passed to xstructured's `with_xstructured_output`, for
example:

```python
structured = agent.with_structured_output(
    OrderReport,
    repair=model,  # ask a model to fix responses that fail validation
)
```

`schema` may be a Pydantic model, a `TypeAdapter`, or named schemas; see the
xstructured documentation for envelopes, multiple values, and streaming.

!!! note

    Agloom's own internal decisions (task analysis, supervisor routing, plans,
    reviews, and loop assessments) use the model's native structured output.
    `with_structured_output` is for the answer your application consumes.

??? example "examples/19_structured_output.py"

    ```python
    --8<-- "examples/19_structured_output.py"
    ```
