# Strategy engine

Agloom's internal LLM analyzer receives the task plus an authoritative inventory
of configured tools and capabilities. It returns a validated `TaskAnalysis`
containing bounded signals such as complexity, estimated steps, decomposition,
parallelism, tool need, validation need, context need, and unavailable required
capabilities. A separate nested `ExecutionBlueprint` carries task-specific
instructions, bounded worker roles, ordered stages, and optional Hybrid
composition. Neither model chooses a topology or grants capabilities. Invalid
structured model output, missing required capabilities, and a selected
delegation topology without workers are explicit failures.

`StrategyEngine.decide(analysis, constraints, policy) -> ExecutionDecision`
chooses DIRECT or one allowed topology. Selection is deterministic for the
default strategy and can be replaced through the documented extension contract.
A one-pattern explicit request bypasses both components. The analyzer is an
internal architectural component rather than a public extension seam. Analyzer
or strategy failures are not fallback signals.
