# Examples

The repository ships 47 runnable examples and seven reference applications.
Every example runs against a live model and asserts facts about the result,
so each one doubles as an end-to-end check of the feature it shows.

## Run the examples

```bash
git clone https://github.com/smuniharish/agloom.git
cd agloom
uv sync
cp .env.example .env   # then set EXPLABS_API_KEY, EXPLABS_BASE_URL, EXPLABS_MODEL
uv run python examples/01_direct.py
```

The examples read an OpenAI-compatible endpoint from the `EXPLABS_*`
variables; `.env` is git-ignored. Examples 44 and 47 also need a PostgreSQL
database.

## Execution

| Example | What it shows |
|---|---|
| [01_direct.py](https://github.com/smuniharish/agloom/blob/master/examples/01_direct.py) | A simple request runs DIRECT: one answer call, no graph, no tools |
| [02_automatic.py](https://github.com/smuniharish/agloom/blob/master/examples/02_automatic.py) | Automatic selection picks the right execution for each request |
| [03_planner.py](https://github.com/smuniharish/agloom/blob/master/examples/03_planner.py) | Planner: independent steps run in parallel waves, then one synthesis |
| [04_react_tools.py](https://github.com/smuniharish/agloom/blob/master/examples/04_react_tools.py) | ReAct: reasoning interleaved with tool calls |
| [05_supervisor.py](https://github.com/smuniharish/agloom/blob/master/examples/05_supervisor.py) | Supervisor: a coordinator delegates to specialist workers |
| [06_pipeline.py](https://github.com/smuniharish/agloom/blob/master/examples/06_pipeline.py) | Pipeline: fixed, ordered stages that build on each other |
| [07_reflection.py](https://github.com/smuniharish/agloom/blob/master/examples/07_reflection.py) | Reflection: draft, review, and revise until approved |
| [08_swarm.py](https://github.com/smuniharish/agloom/blob/master/examples/08_swarm.py) | Swarm: peers hand control to each other |
| [09_blackboard.py](https://github.com/smuniharish/agloom/blob/master/examples/09_blackboard.py) | Blackboard: specialists contribute to a shared board in parallel |
| [10_hybrid.py](https://github.com/smuniharish/agloom/blob/master/examples/10_hybrid.py) | Hybrid: two to four topologies in sequence |
| [11_recursion.py](https://github.com/smuniharish/agloom/blob/master/examples/11_recursion.py) | Planner steps as bounded recursive child executions |
| [12_no_recursion.py](https://github.com/smuniharish/agloom/blob/master/examples/12_no_recursion.py) | The default: Planner steps as plain step agents |

## Integrations and runtime

| Example | What it shows |
|---|---|
| [13_mcp.py](https://github.com/smuniharish/agloom/blob/master/examples/13_mcp.py) | MCP tools, resources, and prompts through an application-owned runtime |
| [14_context.py](https://github.com/smuniharish/agloom/blob/master/examples/14_context.py) | ContextSage middleware summarizes long histories without losing facts |
| [15_feedback.py](https://github.com/smuniharish/agloom/blob/master/examples/15_feedback.py) | Human feedback on a thread reaches the agent's next run |
| [16_behavior.py](https://github.com/smuniharish/agloom/blob/master/examples/16_behavior.py) | BehaviorWeave middleware stops runaway tool use, and the loop obeys |
| [17_refresh.py](https://github.com/smuniharish/agloom/blob/master/examples/17_refresh.py) | refresh-engine re-indexes changed sources when the loop needs it |
| [18_explainability.py](https://github.com/smuniharish/agloom/blob/master/examples/18_explainability.py) | Opt-in explainability with langgraph-xai |
| [19_structured_output.py](https://github.com/smuniharish/agloom/blob/master/examples/19_structured_output.py) | Validated Pydantic objects from any agent |
| [20_hitl.py](https://github.com/smuniharish/agloom/blob/master/examples/20_hitl.py) | Tool-call review with `HumanInTheLoopMiddleware`, and breakpoints on topology nodes, then resume |
| [21_checkpoint_resume.py](https://github.com/smuniharish/agloom/blob/master/examples/21_checkpoint_resume.py) | Conversation memory that survives restarts |
| [22_streaming.py](https://github.com/smuniharish/agloom/blob/master/examples/22_streaming.py) | Model tokens and Agloom events as they happen |
| [23_observability.py](https://github.com/smuniharish/agloom/blob/master/examples/23_observability.py) | Events to logs, metrics, callables, and Langfuse |
| [24_complete_configuration.py](https://github.com/smuniharish/agloom/blob/master/examples/24_complete_configuration.py) | Every `create_agent` parameter in a production-style configuration |

## Loop Engineering

| Example | What it shows |
|---|---|
| [25_loop_automatic_completion.py](https://github.com/smuniharish/agloom/blob/master/examples/25_loop_automatic_completion.py) | One-shot by default; early completion when judged done |
| [26_loop_multi_iteration.py](https://github.com/smuniharish/agloom/blob/master/examples/26_loop_multi_iteration.py) | Iterating until the evaluator confirms the objective |
| [27_loop_progress.py](https://github.com/smuniharish/agloom/blob/master/examples/27_loop_progress.py) | Objective progress measured by the system, not the model |
| [28_loop_retry.py](https://github.com/smuniharish/agloom/blob/master/examples/28_loop_retry.py) | Recovering from a transient failure within a retry budget |
| [29_loop_replan.py](https://github.com/smuniharish/agloom/blob/master/examples/29_loop_replan.py) | Changing approach when a validator rejects the result |
| [30_loop_refresh.py](https://github.com/smuniharish/agloom/blob/master/examples/30_loop_refresh.py) | Refreshing sources when progress stalls |
| [31_loop_stagnation.py](https://github.com/smuniharish/agloom/blob/master/examples/31_loop_stagnation.py) | Stopping early when attempts make no progress |
| [32_loop_budget.py](https://github.com/smuniharish/agloom/blob/master/examples/32_loop_budget.py) | Hard limits on tool calls, iterations, and active time |
| [33_loop_human_intervention.py](https://github.com/smuniharish/agloom/blob/master/examples/33_loop_human_intervention.py) | Pausing the loop for a person and resuming with their decision |
| [34_loop_checkpoint_resume.py](https://github.com/smuniharish/agloom/blob/master/examples/34_loop_checkpoint_resume.py) | A stopped loop continues in a new process |
| [35_loop_recursive.py](https://github.com/smuniharish/agloom/blob/master/examples/35_loop_recursive.py) | Iterations that fan out into bounded child executions |
| [36_loop_topology_adaptation.py](https://github.com/smuniharish/agloom/blob/master/examples/36_loop_topology_adaptation.py) | Switching the execution structure when results fall short |
| [37_loop_feedback.py](https://github.com/smuniharish/agloom/blob/master/examples/37_loop_feedback.py) | Evaluator feedback that steers iterations and is recorded |
| [38_loop_context.py](https://github.com/smuniharish/agloom/blob/master/examples/38_loop_context.py) | The bounded context carried between iterations |
| [39_loop_long_running.py](https://github.com/smuniharish/agloom/blob/master/examples/39_loop_long_running.py) | Streaming progress of a long multi-iteration job |

## Policies, capabilities, and composition

| Example | What it shows |
|---|---|
| [40_policy_allow_deny.py](https://github.com/smuniharish/agloom/blob/master/examples/40_policy_allow_deny.py) | Allowing and denying tool calls with an agentbarrier policy |
| [41_policy_approval.py](https://github.com/smuniharish/agloom/blob/master/examples/41_policy_approval.py) | Approvals: pause for a reviewer, then execute exactly once, or never |
| [42_policy_custom_rules.py](https://github.com/smuniharish/agloom/blob/master/examples/42_policy_custom_rules.py) | Your own rules as Python objects or versioned JSON files |
| [43_policy_mcp.py](https://github.com/smuniharish/agloom/blob/master/examples/43_policy_mcp.py) | Governing MCP tools, resources, and prompts like local tools |
| [44_policy_postgres.py](https://github.com/smuniharish/agloom/blob/master/examples/44_policy_postgres.py) | A PostgreSQL policy store shared by several processes |
| [45_capability_routing.py](https://github.com/smuniharish/agloom/blob/master/examples/45_capability_routing.py) | Capability filters, semantic ranking, and direct capability execution |
| [46_deep_agents.py](https://github.com/smuniharish/agloom/blob/master/examples/46_deep_agents.py) | Agloom and Deep Agents composed in both directions |
| [47_postgres_checkpointer.py](https://github.com/smuniharish/agloom/blob/master/examples/47_postgres_checkpointer.py) | A PostgreSQL checkpointer with thread authorization |

## Reference applications

Complete FastAPI and React applications, each with its own dependencies,
seeded demonstration data, tests, and container configuration. See each
application's README to run it.

| Application | What it demonstrates |
|---|---|
| [Enterprise Knowledge Assistant](https://github.com/smuniharish/agloom/tree/master/examples/enterprise_knowledge_assistant) | Grounded answers with citations over enterprise documents |
| [QA Customer Chatbot](https://github.com/smuniharish/agloom/tree/master/examples/qa_customer_chatbot) | A grounded customer-support agent with operational tools and behavioral guardrails |
| [RCA Generator](https://github.com/smuniharish/agloom/tree/master/examples/rca_generator) | Root-cause analysis from incident events, logs, and metrics |
| [Web Compliance Auditor](https://github.com/smuniharish/agloom/tree/master/examples/web_compliance_auditor) | Auditing a public URL against policies with real MCP servers, PostgreSQL, and pgvector |
| [Engineering Change Investigator](https://github.com/smuniharish/agloom/tree/master/examples/engineering_change_investigator) | Release-readiness assessment from engineering records, live evidence, and pgvector |
| [Autonomous Product Studio](https://github.com/smuniharish/agloom/tree/master/examples/autonomous_product_studio) | Iterative construction and validation with Loop Engineering, approvals, and background jobs |
| [Continuous Evidence Lab](https://github.com/smuniharish/agloom/tree/master/examples/continuous_evidence_lab) | Long-running evidence refresh, claim verification, and adaptive replanning |

The [`examples/observability`](https://github.com/smuniharish/agloom/tree/master/examples/observability)
directory provides the shared Prometheus, Grafana, and Langfuse stack used by
the applications.
