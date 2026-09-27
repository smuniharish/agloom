# Implementation order

The frozen build sequence is: (1) architecture baseline, (2) Python 3.12
packaging and tooling, (3) errors/config/contracts, (4) direct execution,
(5) topology contract and deterministic topology graphs, (6) structured
analyzer/strategy/specification, (7) compiler, (8) runtime/lifecycle/events,
(9) recursion budgets, (10) capabilities and real package adapters,
(11) integration and concurrency tests, (12) docs/examples, (13) benchmark,
security review, clean install, and release build.

No topology is treated as production-ready solely because it is registered:
its behavior must have deterministic tests and a runnable LangGraph path.
External adapters are implemented only after their actual package API is
inspected.
