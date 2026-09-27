# Error reference

Catch the narrowest exception your application can recover from. Every Agloom
exception derives from `AgloomError`.

| Exception | Raised when |
|---|---|
| `ConfigurationError` | Public configuration is invalid or conflicting. |
| `TopologySelectionError` | No permitted execution mode can satisfy the task. |
| `TopologyError` | A topology cannot be constructed or executed. |
| `CapabilityResolutionError` | A named capability is missing, unsupported, or requires async execution. |
| `CapabilityError` | Another capability operation fails. |
| `RecursionLimitError` | A depth, worker, subtask, or execution-time budget is exceeded. |
| `RecursionError` | Recursive execution is disabled or otherwise invalid. |
| `CompilationError` | A selected execution topology cannot be prepared. |
| `LifecycleError` | An invalid runtime lifecycle transition is requested. |
| `ExecutionError` | Execution cannot produce a valid result. |
| `CancellationError` | Cooperative cancellation stops an invocation. |
| `RuntimeError` | Another Agloom runtime failure occurs. |

```python
from agloom import CapabilityResolutionError, ConfigurationError

try:
    result = await agent.ainvoke(task)
except ConfigurationError as error:
    # Report a deployment/configuration problem.
    raise
except CapabilityResolutionError as error:
    # Map an unavailable approved capability to your application response.
    raise
```

Agloom does not use successful-looking fallback values for invalid
configuration or failed execution.
