# Loop Engineering

Bounds, results, and extension contracts of the outer objective loop. See
[Loop Engineering](../concepts/loop-engineering.md) and the
[Loops guide](../guides/loops.md).

## Policy and results

::: agloom.LoopPolicy

::: agloom.LoopResult

::: agloom.LoopState

::: agloom.LoopBudget

::: agloom.LoopHistoryEntry

::: agloom.LoopObservation

::: agloom.LoopEvaluation

::: agloom.LoopDecision

## Enumerations

::: agloom.LoopPhase

::: agloom.LoopAction

::: agloom.LoopTerminationReason

::: agloom.ProgressStatus

::: agloom.ObjectiveStatus

## Extension contracts

::: agloom.LoopObserver

::: agloom.LoopEvaluator

::: agloom.AsyncLoopEvaluator

::: agloom.LoopController

## Built-in components

::: agloom.DefaultLoopObserver

::: agloom.DefaultLoopEvaluator

::: agloom.ModelLoopEvaluator

::: agloom.LoopAssessment

::: agloom.DefaultLoopController

::: agloom.IterationFailure

## Ecosystem signals

::: agloom.behavior_signals

::: agloom.LOOP_INTERRUPT

The `type` of the interrupt payload raised when the loop waits for a human
(`WAIT` or `HUMAN_INTERVENTION`) on an agent with a checkpointer. The payload
also carries `loop_id`, `iteration`, `action`, and `reason`. Resume with
[`Agent.resume`][agloom.Agent.resume], passing human feedback text or
`{"feedback": ...}` as the value.
