# State model

| State | Owner | Lifetime |
|---|---|---|
| Task input | Invocation | Immutable request |
| Analysis/decision/specification | Factory resolution | Construction/dispatch |
| Topology state | LangGraph | One graph thread/run |
| Runtime state | Agent runtime | Per invocation |
| Worker state | Topology graph | Per worker task |
| Recursion budget | Runtime invocation | Parent and all descendants |
| Lifecycle state | Runtime invocation | Created through terminal state |

Graph state is an explicit mapping with a fresh schema per graph. Blackboard
state is versioned and updated through conflict-checked reducers; no global
mutable workspace is allowed. Child tasks receive explicit values and return
explicit results. Checkpoint persistence is delegated to the LangGraph
checkpointer supplied by the caller.
