# Agent

Create agents with [`create_agent`][agloom.create_agent]. The returned
[`Agent`][agloom.Agent] is a LangChain `Runnable`: use it anywhere a runnable is
accepted, including as a node or sub-agent of another graph; see
[Agents inside another graph](../guides/human-in-the-loop.md#agents-inside-another-graph)
for pausing there.

See the [configuration guide](../guides/configuration.md) for a walkthrough of
every parameter.

::: agloom.create_agent

::: agloom.Agent
    options:
      merge_init_into_class: false

::: agloom.BREAKPOINT_INTERRUPT

The `type` of the interrupt reported when a run stops at an `interrupt_before`
or `interrupt_after` node. Its payload lists the configured
`interrupt_before` and `interrupt_after` node names. Continue the thread with
[`Agent.resume`][agloom.Agent.resume]. See
[Human-in-the-loop and resume](../guides/human-in-the-loop.md).
