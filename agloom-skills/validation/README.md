# Agloom Agent Skill validation

This directory defines repeatable validation for the canonical Agloom Agent
Skill. It is not a runtime test suite or a host-specific package.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

## Structural validation

For every change:

1. Confirm [`../skills/agloom/SKILL.md`](../skills/agloom/SKILL.md) exists and
   starts with YAML frontmatter.
2. Confirm `name` is exactly `agloom`, matches the directory, contains only
   lowercase letters and hyphens, and is at most 64 characters.
3. Confirm `description` is non-empty, no more than 1,024 characters, and
   states what the skill enables and when it activates.
4. Confirm frontmatter contains only `name` and `description`.
5. Resolve every relative Markdown target in the distribution.
6. Confirm there is one canonical `skills/agloom/` source and no duplicated
   Claude, Codex, Cursor, or Copilot copy.
7. Confirm `SKILL.md` and every reference link to current examples and
   documentation.
8. Search for stale names, invented APIs or commands, credentials, private
   imports, and the incorrect “nine topologies” terminology.

## Source-accuracy review

| Claim area | Source of truth |
| --- | --- |
| Public imports | [`src/agloom/__init__.py`](https://github.com/smuniharish/agloom/blob/master/src/agloom/__init__.py) |
| Agent construction | [`src/agloom/agent/factory.py`](https://github.com/smuniharish/agloom/blob/master/src/agloom/agent/factory.py) |
| Agent methods | [`src/agloom/agent/handle.py`](https://github.com/smuniharish/agloom/blob/master/src/agloom/agent/handle.py) |
| Configuration models | [`src/agloom/models.py`](https://github.com/smuniharish/agloom/blob/master/src/agloom/models.py) |
| Dependencies and Python support | [`pyproject.toml`](https://github.com/smuniharish/agloom/blob/master/pyproject.toml) |
| Public behavior | [Agloom documentation](https://agloom.readthedocs.io/en/latest/) |
| Executable integrations | [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples) |
| Regression behavior | [`tests/`](https://github.com/smuniharish/agloom/tree/master/tests) |

Omit any behavior that lacks implementation, a focused test, an executable
example, or authoritative documentation.

## Agent-task matrix

| Task | Activates | Grounded route | Avoids |
| --- | --- | --- | --- |
| “Automatically choose an Agloom topology for each task.” | Yes | Selection rules and automatic example | Inventing a second router |
| “Force Planner for this workflow.” | Yes | One explicit pattern | Unnecessary analysis |
| “Allow only Planner or Supervisor.” | Yes | Pattern allow-list | Treating the list as Hybrid |
| “Compose ReAct and Reflection.” | Yes | Explicit Hybrid composition | Calling it constrained selection |
| “Add local and MCP capabilities.” | Yes | Capability routing and MCP example | Exposing every remote tool blindly |
| “Bound recursive child work.” | Yes | `RecursionPolicy` | Unbounded autonomous loops |
| “Pause for approval and resume.” | Yes | Checkpointer, authorizer, and HITL example | Treating a thread ID as authorization |
| “Trace runs in Langfuse and Grafana.” | Yes | Observer and observability example | Swallowing telemetry failures |
| “Debug a raw LangGraph graph with no Agloom usage.” | No | Use LangGraph guidance | Claiming Agloom owns unrelated graphs |
| “Autoscale Kubernetes workers.” | No | Use deployment infrastructure | Misrepresenting orchestration scaling |

## Repository validation

Skill-only work must pass structural, link, source-accuracy, activation-matrix,
and diff review. If runtime files change, run the repository formatter, linter,
type checker, focused/full tests as appropriate, and strict MkDocs build.
