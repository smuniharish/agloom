# Responsibility matrix

| Concern | Owner | Not owned by |
|---|---|---|
| Public construction / validation | Agent factory | Topologies |
| Task structure and capability requirements | Internal LLM analyzer | Runtime/compiler |
| Permissible execution choice | Strategy engine | Analyzer |
| Compiler-ready selection | ArchitectureSpec | Raw LLM output |
| Graph shape | Topology builder | Capability adapter |
| Graph compilation | Compiler | Agent factory |
| Invocation / lifecycle / cancellation | Runtime | Topology |
| Child budgets / recursion | Runtime recursion coordinator | Worker |
| Checkpoint persistence | LangGraph + caller checkpointer | Agloom database |
| Provider behavior | LangChain model implementation | Agloom strategy |
| Specialized capability behavior | Named ecosystem package | Duplicate Agloom subsystem |
