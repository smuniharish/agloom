# Capability API

Use these contracts when the `tools=` and `capabilities=` shortcuts are not
enough. Registries and routers are scoped to an agent; Agloom does not create a
global capability catalog.

## `CapabilityDescriptor`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | **Required** | — | Stable, non-empty capability name. |
| `description` | `str` | **Required** | — | Human-readable purpose used during selection. |
| `kind` | `CapabilityKind` | **Required** | — | `TOOL`, `APPLICATION`, or `MCP`. |
| `value` | `Any` | **Required** | — | Object that implements the capability. Hidden from object representations. |
| `provider` | `str` | Optional | `"application"` | Non-empty source identifier. |
| `metadata` | `Mapping[str, Any]` | Optional | `{}` | Immutable metadata copied at construction. |

## `InMemoryCapabilityRegistry`

Constructor:

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `descriptors` | `Iterable[CapabilityDescriptor]` | Optional | `()` | Initial descriptors. Duplicate names are rejected. |

Methods:

| Method | Description |
|---|---|
| `register(descriptor)` | Registers one descriptor and rejects duplicate names. |
| `descriptor(name)` | Returns the descriptor with the exact stable name. |
| `descriptors()` | Returns a stable tuple snapshot. |
| `resolve(name)` | Returns the descriptor's implementation value. |
| `names` | Returns registered names as a tuple. |

Implement `CapabilityRegistry` when storage needs differ.

## `DefaultCapabilityRouter`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `policy` | `CapabilityPolicy \| None` | Optional | `None` | Authorization/filter policy; defaults to `AllowAllCapabilities`. |
| `embeddings` | `Embeddings \| None` | Optional | `None` | Semantic matching through the LangChain embeddings contract. |
| `retriever` | `BaseRetriever \| None` | Optional | `None` | Retriever returning documents with `capability_name` metadata. |
| `reranker` | `BaseDocumentCompressor \| None` | Optional | `None` | Reranks selected capability documents. |
| `max_results` | `int` | Optional | `20` | Maximum returned capabilities; must be at least 1. |

Implement `CapabilityPolicy.allows(task, capability)` to enforce application
authorization before relevance selection. Implement `CapabilityRouter.route`
and `CapabilityRouter.aroute` when you need a complete custom routing
algorithm.

## `CapabilitySelection`

| Member | Type | Description |
|---|---|---|
| `task` | `str` | Task used for routing. |
| `capabilities` | `tuple[CapabilityDescriptor, ...]` | Selected descriptors in routing order. |
| `names` | `tuple[str, ...]` | Selected stable names. |
| `tools` | `tuple[object, ...]` | Selected values whose kind is `TOOL`. |

See [Capabilities](../concepts/capabilities.md) for application and MCP
integration examples.
