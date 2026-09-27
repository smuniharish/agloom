# Validation report

Validation was run locally on Windows with CPython 3.12.14.

## Observed dependency versions

| Package | Version |
|---|---:|
| LangChain | 1.4.2 |
| LangChain Core | 1.6.5 |
| LangGraph | 1.2.12 |
| Pydantic | 2.13.5 |
| `feedback-manager` | 0.1.0 |
| `behaviorweave` | 0.1.0 |
| `contextsage` | 0.1.0 |
| `mcp-capability-router` | 0.1.0 |
| `refresh-engine` | 0.1.0 |
| `langgraph-xai` | 0.1.0 |
| `xstructured` | 0.1.0 |

The seven named capability packages are required runtime dependencies. Tests
exercise their public APIs and the constructor overrides exposed through
`create_agent`. LangSmith is not configured or required, though it is present
transitively in the LangChain dependency tree.

## Quality gates

| Check | Result |
|---|---|
| `uv --system-certs sync --python 3.12 --group dev` | Passed |
| `uv --system-certs run --python 3.12 ruff check .` | Passed |
| `uv --system-certs run --python 3.12 black --check .` | Passed |
| `uv --system-certs run --python 3.12 pyrefly check` | Passed; zero errors |
| `uv --system-certs run --python 3.12 pytest -q` | **59 passed** |
| `uv --system-certs run --python 3.12 mkdocs build --strict` | Passed; no broken-page warnings |
| `uv --system-certs build --out-dir dist` | Built wheel and source archive |
| Clean Python 3.12.14 wheel install and direct invocation | Passed |
| 22 example scripts | Passed with deterministic models and required packages |

## Clean installation

The built wheel installed into a new Python 3.12.14 environment with all
required dependencies. The environment successfully imported
`from agloom import create_agent` and invoked a deterministic LangChain model.

## DIRECT performance check

The benchmark harness was run for five iterations per case. Results below are
local microbenchmarks using deterministic LangChain fake models and include
agent construction and invocation; they are not provider latency estimates.

| Case | Median ms | Peak KiB | Topology compiles |
|---|---:|---:|---:|
| DIRECT greeting | 3.749 | 174.4 | 0 |
| Automatic Planner selection | 10.143 | 217.3 | 5 |
| Explicit Planner | 6.633 | 119.6 | 5 |
| Supervisor | 7.300 | 125.2 | 5 |
| Blackboard with two workers | 9.532 | 146.4 | 5 |
| Hybrid | 25.035 | 430.3 | 5 |
| Recursive Planner | 10.318 | 173.4 | 5 |

The measurable DIRECT invariant holds: trivial requests compiled no topology.

## Security review

The security review found two high-severity boundary concerns, both addressed:

1. Checkpoint runs, resumes, and state reads require a `checkpoint_authorizer`
   callback that checks the requested `thread_id` against the application's
   authenticated principal.
2. Untrusted message lists accept only user/assistant roles; system, developer,
   and tool messages are rejected before model invocation. Trusted instructions
   are configured through `system_prompt`.

Regression tests cover authorization and message-role validation.

## Scope and limitations

- Provider-backed inference and a live MCP server were not exercised; examples
  use deterministic models or an empty MCP client. External service behavior
  therefore remains application-specific.
- Performance samples are local microbenchmarks, not load, throughput, or
  long-duration production capacity tests.
- The GitHub Actions workflow was not run by a remote CI service during this
  validation.
- Local tools, application services, and MCP entries are registered in an
  abstract agent-local capability catalog. Policy, embedding, retrieval, and
  reranking tests use LangChain contracts; Agloom does not replace or duplicate
  package-specific service lifecycles.
