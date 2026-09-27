# Error model

`AgloomError` is the public base. Configuration, selection, topology,
capability, recursion, compilation, lifecycle, cancellation, and execution
errors use typed subclasses. Errors preserve the original exception with
exception chaining and include safe identifiers and relevant limit/context.

Invalid analyzer output, unavailable required capabilities, unsupported
topologies, capability failures, tool/worker/model failures, checkpoint
failures, timeouts, and cancellation are visible failures. No broad
catch-and-continue behavior is implicit. A topology may continue after a worker
failure only under an explicit configured failure policy.

Untrusted system/developer/tool messages and unauthorized checkpoint thread
access raise `ConfigurationError` before the model or checkpointer receives the
request.
