# Errors

Every Agloom exception derives from [`AgloomError`][agloom.AgloomError].
LangGraph interrupts and agentbarrier runtime exceptions (for example
`agentbarrier.errors.ApprovalRequired`) are never wrapped. See
[Errors](../concepts/errors.md) for the hierarchy and handling guidance.

::: agloom.AgloomError

## Configuration

::: agloom.ConfigurationError

::: agloom.PolicyConfigurationError

## Execution

::: agloom.ExecutionPaused

::: agloom.HarnessError

::: agloom.ExecutionError

::: agloom.CancellationError

::: agloom.RecursionLimitError

::: agloom.PolicyViolationError

## Selection and compilation

::: agloom.TopologyError

::: agloom.TopologySelectionError

::: agloom.CompilationError

## Capabilities

::: agloom.CapabilityError

::: agloom.CapabilityResolutionError

## Loop Engineering

::: agloom.LoopError

::: agloom.LoopBudgetExceededError

::: agloom.LoopDecisionError

::: agloom.LoopEvaluationError

::: agloom.LoopStateError
