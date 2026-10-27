# Capabilities and MCP

A capability is anything the agent can use: a LangChain tool, an application
service, or a tool, resource, or prompt exposed by an MCP server. Agloom keeps
all of them in **one catalog** and routes the relevant subset to each request,
so task analysis, the model, and policies all see the same, focused inventory.

![Capability routing](../assets/diagrams/capability-routing-light.png#only-light)
![Capability routing](../assets/diagrams/capability-routing-dark.png#only-dark)

## The catalog

| Source | Configured with | Kind | Provider | The model can call it |
|---|---|---|---|---|
| LangChain tools and annotated functions | `tools=[...]` | `tool` | `langchain` | Yes |
| Tools contributed by middleware | `middleware=[...]` | `tool` | `middleware` | Yes |
| Application services | `capabilities={"name": service}` | `application` | `application` | No; call it from code |
| MCP tools | `mcp=runtime` | `mcp` | `mcp-capability-router` | Yes |
| MCP resources and prompts | `mcp=runtime` | `mcp` | `mcp-capability-router` | No; read or render from code |

Each entry is a [`CapabilityDescriptor`][agloom.CapabilityDescriptor] with a
unique name, a description used for routing and analysis, its kind and
provider, and the implementing object. Tool names must be unique across
`tools=` and middleware. An application service's description comes from its
`description` attribute.

Tools that LangChain middleware contributes (`AgentMiddleware.tools`), such as
`write_todos` from `TodoListMiddleware` or the filesystem tools of Deep Agents'
`FilesystemMiddleware`, are registered automatically. They are routed and shown
to task analysis like your own tools, hidden when the policy denies them for
every call, and can be declared consequential in the
[`PolicyManager`][agloom.PolicyManager]. With a policy, they need allow rules
like any other tool.

Application capabilities, MCP resources, and MCP prompts appear in routing and
task analysis, and your code executes them through the agent so that the
policy still applies. An application capability is any object with a
`description` and an `invoke` method (or a plain callable):

```python
class PolicyStore:
    description = "Support policies for returns, refunds, and damaged goods."

    def invoke(self, request: dict) -> dict:
        return {"topic": request["topic"], "policy": "Damaged goods are refunded in full."}


agent = create_agent(
    model=model, tools=[order_status], capabilities={"policy_store": PolicyStore()}
)
print(agent.route_capabilities("Find the refund policy for damaged goods.").names)
print(agent.execute_capability("policy_store", {"topic": "damaged goods"}))
```

## Routing

Each request is routed before analysis and execution by a
[`CapabilityRouter`][agloom.CapabilityRouter]; the default is
[`DefaultCapabilityRouter`][agloom.DefaultCapabilityRouter]:

1. **Filter.** A [`CapabilityFilter`][agloom.CapabilityFilter] decides which
   capabilities may be considered for this task. Rejected capabilities are
   never shown to the analyzer or the model.
2. **Select.** Without ranking components, every remaining capability is
   selected. Nothing is silently truncated.
3. **Rank** (optional). Capabilities named in the task (by name or `aliases`
   metadata) come first, then semantic matches from an in-memory vector index
   built with your `embeddings`, then documents from your `retriever` that
   carry `capability_name` metadata. An optional `reranker` (any LangChain
   document compressor) reorders the candidates, and at most `max_results`
   (default 20) are selected.

```python
from agloom import CapabilityFilter, DefaultCapabilityRouter, create_agent


class TenantCapabilities(CapabilityFilter):
    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed

    def allows(self, task, capability) -> bool:
        return capability.name in self.allowed


router = DefaultCapabilityRouter(
    capability_filter=TenantCapabilities({"order_status", "search_docs"}),
    embeddings=embeddings,  # any LangChain Embeddings
    max_results=8,
)
agent = create_agent(model=model, tools=tools, capability_router=router)
```

The model sees only the routed tools, and routing is enforced at execution
time: if the model calls one of the agent's tools that was not offered for the
current request, the call is answered with an error tool message (`Tool 'name'
is not available for this request.`) and a `ToolCallDenied` event with reason
`ToolNotOffered`; the tool does not run. DIRECT execution offers no tools at
all.

### When no capability fits

A request is never rejected because a capability is missing. During automatic
selection, task analysis lists the actions the user asked for that no routed
tool can perform. The run proceeds: the execution is told which actions are
unavailable and must not be claimed, so the answer says what cannot be done,
and the `TopologySelected` event reports how many were missing. See
[Execution selection](execution-selection.md#task-analysis).
[`CapabilityResolutionError`][agloom.CapabilityResolutionError] is raised only
when your code executes a capability that is not registered or cannot be
executed, or, under a policy, calls a tool with arguments that do not match
its schema.

!!! note "Routing is not authorization"

    A filter decides what the model is **offered** for a task; it is not an
    access-control mechanism. Authorization belongs to the
    [`PolicyManager`][agloom.PolicyManager], which decides whether a specific
    call with specific arguments may run, needs approval, or must run exactly
    once, including calls your own code makes with `execute_capability`. Use
    both: tools that a policy denies for every call are also hidden from the
    model.

## MCP servers

Agloom connects to MCP through
[mcp-capability-router](https://pypi.org/project/mcp-capability-router/)
(FastMCP based). Your application owns the `MCPRuntime`: it registers servers,
manages their lifecycle, and passes the runtime to the agent.

```python
import asyncio

from fastmcp import FastMCP
from mcp_capability_router import MCPRuntime, RefreshPolicy

from agloom import create_agent

inventory = FastMCP("inventory")


@inventory.tool
def stock_level(sku: str) -> int:
    """Return the number of units in stock for a SKU."""
    return {"SKU-1": 12}.get(sku, 0)


async def main() -> None:
    async with MCPRuntime() as runtime:
        await runtime.register_mcp(
            "inventory", inventory, refresh=RefreshPolicy(on_register=True)
        )
        agent = create_agent(model=model, mcp=runtime)
        answer = await agent.ainvoke("How many units of SKU-1 are in stock?")
        print(answer.content)


asyncio.run(main())
```

!!! warning "Always register servers with a refresh policy"

    `register_mcp` does not discover a server's capabilities by itself. Pass
    `refresh=RefreshPolicy(on_register=True)`, or call `await runtime.refresh()`
    after registering; otherwise the agent silently sees no MCP tools,
    resources, or prompts. Add `interval=` or `change_events=` to the policy to
    keep the catalog current. Agloom reads the runtime's catalog on every
    request, so refreshed capabilities are picked up without recreating the
    agent.

- **Targets.** `register_mcp(server_id, target)` accepts a URL, a script
  path, an MCP configuration dictionary, an in-process `FastMCP` server, a
  transport, or a `fastmcp.Client`.
- **Stdio servers.** Register them with an MCP configuration and set
  `"keep_alive": False`; mcp-capability-router 0.2.0 otherwise keeps stdio
  server processes running after the runtime closes:

    ```python
    await runtime.register_mcp(
        "ops",
        {
            "mcpServers": {
                "ops": {
                    "command": "python",
                    "args": ["ops_server.py"],
                    "transport": "stdio",
                    "keep_alive": False,
                }
            }
        },
        refresh=RefreshPolicy(on_register=True),
    )
    ```

- **Names.** MCP tools are named `<server_id>__<tool>` (for example
  `inventory__stock_level`), with unsupported characters replaced by `_` and
  names longer than 64 characters shortened with a stable digest. Use these
  names in policies. Resources and prompts are named `mcp:<capability_id>`.
- **Async only.** Agents with `mcp=` use `ainvoke`, `ainvoke_loop`,
  `astream`, `aresume`, `aroute_capabilities`, and `aexecute_capability`.

??? example "examples/13_mcp.py"

    ```python
    --8<-- "examples/13_mcp.py"
    ```

## Inspecting and executing capabilities

| Method | Returns |
|---|---|
| `agent.capabilities` | The [`CapabilityRegistry`][agloom.CapabilityRegistry] of local tools, middleware tools, and application services |
| `agent.route_capabilities(task)` | A [`CapabilitySelection`][agloom.CapabilitySelection] (`.names`, `.tool_names`, `.capabilities`) |
| `await agent.aroute_capabilities(task)` | The same, including MCP capabilities |
| `agent.execute_capability(name, input)` | What one local or application capability produces (a tool's output or the service's return value), under the policy |
| `await agent.aexecute_capability(name, input)` | The same, including MCP capabilities, whose results come from mcp-capability-router unchanged: a LangChain `ToolMessage` for a tool, the list of MCP resource contents for a resource, and MCP's `GetPromptResult` for a prompt |

Executing a capability directly goes through the same policy as a model tool
call: a tool's arguments are validated against its schema first (invalid ones
raise [`CapabilityResolutionError`][agloom.CapabilityResolutionError]), denied
calls raise [`PolicyViolationError`][agloom.PolicyViolationError] (agentbarrier's
`PolicyDenied` for a consequential capability) and emit `ToolCallDenied`, and
consequential capabilities execute exactly once per business key.

??? example "examples/45_capability_routing.py"

    ```python
    --8<-- "examples/45_capability_routing.py"
    ```
