# Recursion policy

`RecursionPolicy` is immutable and validates positive bounded values for
`max_depth`, `max_workers`, `max_subtasks`, and `max_execution_time`, plus an
optional allow-list of nested topologies. Recursion defaults off. A shared
per-invocation budget reserves task and worker capacity before dispatch and
releases worker capacity in `finally` paths. Time limits are monotonic deadlines,
not wall-clock timestamps.

Exceeding a limit raises a specific `RecursionLimitError` with the limit and
observed value. No implicit retry, expansion, or policy relaxation is permitted.
The policy is enforced by the runtime, not delegated to topology implementations.
