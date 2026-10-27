# Streaming

`agent.stream` and `agent.astream` expose LangGraph's stream modes for the
whole run: the outer loop and the selected execution, including nested
topology graphs.

| You want | Use | Each item |
|---|---|---|
| Model tokens as they are generated | `stream_mode="messages"` | `(token, metadata)` |
| Agloom decisions as they happen (routing, selection, policy, loop) | `stream_mode="custom"` | an event dictionary |
| Both | `stream_mode=["messages", "custom"]` | `(mode, chunk)` |
| Loop progress | `stream_mode="updates"` (the default) | a step update of the top-level graph |

`messages` and `custom` always include nested topology graphs; other modes
report the top-level graph unless you pass `subgraphs=True`.

## Tokens

```python
from langchain_core.messages import AIMessageChunk

for token, metadata in agent.stream("Where is order A-1001?", stream_mode="messages"):
    if isinstance(token, AIMessageChunk):
        print(token.text, end="", flush=True)
```

The stream contains the tokens of every model call that produces work or
answers, plus tool results, from every worker of the selected topology.
Agloom's internal structured decisions (task analysis, routing decisions,
plans, reviews, and loop assessments) are never streamed. Use `metadata` to
tell agents apart, for example by `langgraph_node`.

## Agloom events

`stream_mode="custom"` yields one JSON-compatible dictionary per
[runtime event](../concepts/observability.md#event-catalog) emitted inside the
run: the event name under `agloom_event`, the `run_id`, and the event's payload
fields, plus `topology` for the events that report one. Forward them to a
client as they arrive, for example as server-sent events:

```python
import json


def progress(task: str):
    for event in agent.stream(task, stream_mode="custom"):
        yield f"data: {json.dumps(event)}\n\n"
```

Events emitted before and after the run (`ExecutionStarted` and the final
`ExecutionCompleted`, `ExecutionPaused`, `ExecutionFailed`, or
`ExecutionCancelled`) are delivered to [observers](../concepts/observability.md#observers),
which receive every event as a typed [`AgentEvent`][agloom.AgentEvent]. Use an
observer when your code needs to react to specific events.

## Tokens and events together

With several modes, each item is a `(mode, chunk)` pair:

```python
for mode, chunk in agent.stream(
    "Where is order A-1001?", stream_mode=["messages", "custom"]
):
    if mode == "messages":
        token, metadata = chunk
        if isinstance(token, AIMessageChunk):
            print(token.text, end="", flush=True)
    else:
        print("\n[event]", json.dumps(chunk))
```

Pass `subgraphs=True` to also receive LangGraph namespaces, as in LangGraph:
items become `(namespace, chunk)` or `(namespace, mode, chunk)`, and every mode
includes the nested graphs.

## Async

```python
async for token, metadata in agent.astream(
    "Where is order A-1001?", stream_mode="messages"
):
    ...
```

Agents configured with `mcp=`, `refresh=`, or `feedback=` stream with
`astream` only.

## Inside LCEL compositions

An agent is a LangChain runnable, so it composes with prompts, parsers, and
other runnables. When you stream a composition, the agent emits its final
answer as one chunk; stream the agent itself with `stream_mode="messages"`
when you need its tokens.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda request: request["question"]) | agent | StrOutputParser()
print(chain.invoke({"question": "Where is order A-1002?"}))
```

## Streaming with checkpoints

With a checkpointer, pass the thread in `config` as usual. A stream that
reaches an interrupt (an approval, a reconciliation, a tool-call review, a
breakpoint, or a loop pause) simply ends; inspect the pending interrupts and
resume:

```python
config = {"configurable": {"thread_id": "ticket-4711"}}
for _ in agent.stream("Refund order A-1042 in full.", config, stream_mode="custom"):
    pass
pending = agent.get_state(config).interrupts
if pending:
    print(pending[0].value)
```

Additional keyword arguments of `stream` are passed to LangGraph, and
`context=` passes LangGraph runtime context to tools and middleware.

??? example "examples/22_streaming.py"

    ```python
    --8<-- "examples/22_streaming.py"
    ```
