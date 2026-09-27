# Testing strategy

Unit tests validate contracts, configuration, analysis, decisions, each topology
builder, reducers, lifecycle, recursion budgets, and capability adapters.
Integration tests exercise actual LangGraph invocation, events, streaming,
checkpointer/interrupt paths where supported, and the public API using fake
LangChain-compatible models/tools without network calls.

Failure tests assert preserved causes and typed errors. Isolation/concurrency
tests execute multiple handles and invocations. Performance tests compare direct
construction with graph compilation and assert no direct compiler call; timing
numbers are informative, not flaky correctness thresholds. Optional-package
tests run with the relevant extras installed and otherwise explicitly skip.
Architecture tests assert exactly eight topology identities and terminology.
