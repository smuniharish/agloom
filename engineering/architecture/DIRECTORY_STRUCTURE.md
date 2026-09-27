# Directory structure

```text
src/agloom/
  agent/          public factory and runnable agent handle
  analysis/       structured task analysis
  capabilities/   abstract catalog, providers, router, and integrations
  compiler/       architecture validation and LangGraph compilation
  runtime/        events, lifecycle, cancellation controller, recursion
  strategy/       DIRECT/topology selection
  topology/       shared contracts and one module per built-in topology
  __init__.py     curated public API
  errors.py       typed framework errors
  models.py       configuration and execution contracts
tests/
  test_agent.py
  test_capabilities.py
  test_integrations.py
docs/
  architecture/ guides/ concepts/ api/ examples/ development/
examples/         runnable usage examples
benchmarks/       local performance harness
```

Policies and state contracts currently live in `models.py`, `runtime/`, and
`topology/base.py`; they are not separate packages. Each built-in topology has
its own module (`react.py`, `supervisor.py`, `pipeline.py`, `planner.py`,
`reflection.py`, `swarm.py`, `blackboard.py`, and `hybrid.py`); shared builder
helpers live in `topology/_shared.py`, while `topology/builtins.py` holds their
registry. Modules are organized by responsibility, and public imports are
curated in `agloom.__init__`.
