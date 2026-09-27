# Extensibility

Supported public extension points include `Strategy`, `ArchitectureCompiler`,
`CapabilityRegistry`, `CapabilityRouter`, `CapabilityPolicy`, capability
providers, observers, and application policies. Each has a concrete default or
a documented injection point.

Prefer the existing LangChain contracts for models, tools, embeddings,
retrievers, rerankers, callbacks, and messages. Prefer LangGraph contracts for
checkpointers, interrupts, commands, and graph execution. This keeps custom
components interoperable instead of creating Agloom-specific duplicates.

Extension failures remain explicit. A custom router cannot be mixed with
default-router component options, and checkpoint configuration requires an
authorizer.
