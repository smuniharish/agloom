# Examples

Runnable examples for every Agloom feature, and seven complete reference
applications. Every script runs against a real OpenAI-compatible chat model and
checks the facts in its answer, so a passing run shows the feature working end
to end.

## Run the scripts

1. Install the development environment from the repository root. The default
   dependency groups include everything the examples need:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and set `EXPLABS_API_KEY`, `EXPLABS_BASE_URL`,
   and `EXPLABS_MODEL` for any OpenAI-compatible endpoint that supports tool
   calling and structured output. The examples read `.env`; variables already
   set in the environment take precedence.

3. Run any example:

   ```bash
   uv run python examples/01_direct.py
   ```

`44_policy_postgres.py` and `47_postgres_checkpointer.py` also need a
PostgreSQL URL in `AGLOOM_POSTGRES_URL`. `23_observability.py` sends traces to
Langfuse when `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_BASE_URL` are set.

To run every script as a test, use `uv run pytest tests/test_examples.py -m network`.

## Execution modes and topologies

| Example | Shows |
|---|---|
| [`01_direct.py`](01_direct.py) | DIRECT execution: a simple request is answered by one model call, with no graph and no tools. |
| [`02_automatic.py`](02_automatic.py) | Automatic execution selection: one agent, the right execution for each request. |
| [`03_planner.py`](03_planner.py) | Planner: independent steps run in parallel waves, then one synthesis. |
| [`04_react_tools.py`](04_react_tools.py) | ReAct: a tool-calling agent that reasons, acts, and observes. |
| [`05_supervisor.py`](05_supervisor.py) | Supervisor: a coordinator delegates to specialist workers until the job is done. |
| [`06_pipeline.py`](06_pipeline.py) | Pipeline: fixed, ordered stages that each build on the previous output. |
| [`07_reflection.py`](07_reflection.py) | Reflection: draft, review against the request, revise until approved. |
| [`08_swarm.py`](08_swarm.py) | Swarm: peers hand control to each other without a central coordinator. |
| [`09_blackboard.py`](09_blackboard.py) | Blackboard: specialists contribute to a shared workspace in parallel. |
| [`10_hybrid.py`](10_hybrid.py) | Hybrid: compose two to four topologies in sequence. |
| [`11_recursion.py`](11_recursion.py) | Recursion: Planner steps run as bounded recursive child executions. |
| [`12_no_recursion.py`](12_no_recursion.py) | No recursion: the default keeps Planner steps as plain step agents. |

## Integrations

| Example | Shows |
|---|---|
| [`13_mcp.py`](13_mcp.py) | MCP: tools, resources, and prompts from MCP servers through an app-owned runtime. |
| [`14_context.py`](14_context.py) | Context management: ContextSage summarizes long histories without losing facts. |
| [`15_feedback.py`](15_feedback.py) | Human feedback: corrections on a thread reach the agent's next run. |
| [`16_behavior.py`](16_behavior.py) | Behavior guardrails: BehaviorWeave stops runaway tool use and the loop obeys. |
| [`17_refresh.py`](17_refresh.py) | Fresh sources: refresh-engine re-indexes changed data when the loop needs it. |
| [`18_explainability.py`](18_explainability.py) | Explainability: opt-in capture of what the agent did, for audit and explanation. |
| [`19_structured_output.py`](19_structured_output.py) | Structured output: validated Pydantic objects from any Agloom agent. |
| [`45_capability_routing.py`](45_capability_routing.py) | Capability routing: show each request only the capabilities it needs. |
| [`46_deep_agents.py`](46_deep_agents.py) | Deep Agents and Agloom, in both directions. |

## Persistence, people, and operations

| Example | Shows |
|---|---|
| [`20_hitl.py`](20_hitl.py) | Human in the loop: review tool calls before they run; breakpoints for debugging. |
| [`21_checkpoint_resume.py`](21_checkpoint_resume.py) | Checkpoints: conversation memory that survives restarts. |
| [`22_streaming.py`](22_streaming.py) | Streaming: model tokens and Agloom events as they happen. |
| [`23_observability.py`](23_observability.py) | Observability: events to logs, metrics, plain callables, and Langfuse. |
| [`24_complete_configuration.py`](24_complete_configuration.py) | A complete, production-style configuration in one place. |
| [`47_postgres_checkpointer.py`](47_postgres_checkpointer.py) | PostgreSQL checkpointer: durable threads that any worker can resume. |

## Loop Engineering

| Example | Shows |
|---|---|
| [`25_loop_automatic_completion.py`](25_loop_automatic_completion.py) | One-shot by default, early completion when judged done. |
| [`26_loop_multi_iteration.py`](26_loop_multi_iteration.py) | Keep working until the evaluator confirms the objective. |
| [`27_loop_progress.py`](27_loop_progress.py) | Measure objective progress from the system, not from the model. |
| [`28_loop_retry.py`](28_loop_retry.py) | Recover from a transient failure within a retry budget. |
| [`29_loop_replan.py`](29_loop_replan.py) | Change the approach when a validator rejects the result. |
| [`30_loop_refresh.py`](30_loop_refresh.py) | When progress stalls, refresh sources before giving up. |
| [`31_loop_stagnation.py`](31_loop_stagnation.py) | Stop early when repeated attempts make no progress. |
| [`32_loop_budget.py`](32_loop_budget.py) | Limits on tool calls, iterations, and active time. |
| [`33_loop_human_intervention.py`](33_loop_human_intervention.py) | Pause the loop for a person, resume with their decision. |
| [`34_loop_checkpoint_resume.py`](34_loop_checkpoint_resume.py) | A stopped loop continues in a new process where it left off. |
| [`35_loop_recursive.py`](35_loop_recursive.py) | Each iteration may fan out into bounded child executions. |
| [`36_loop_topology_adaptation.py`](36_loop_topology_adaptation.py) | Switch the execution structure when results fall short. |
| [`37_loop_feedback.py`](37_loop_feedback.py) | Evaluator feedback steers iterations and is recorded for review. |
| [`38_loop_context.py`](38_loop_context.py) | Each iteration sees the previous result, within bounds. |
| [`39_loop_long_running.py`](39_loop_long_running.py) | Stream progress while a multi-iteration job runs. |

## Policy Manager

| Example | Shows |
|---|---|
| [`40_policy_allow_deny.py`](40_policy_allow_deny.py) | Allow and deny tool calls with an agentbarrier runtime policy. |
| [`41_policy_approval.py`](41_policy_approval.py) | Pause for a human, then execute exactly once, or never. |
| [`42_policy_custom_rules.py`](42_policy_custom_rules.py) | Bring your own rules as Python objects or versioned JSON files ([`policies/`](policies/)). |
| [`43_policy_mcp.py`](43_policy_mcp.py) | Govern MCP tools, resources, and prompts like local tools. |
| [`44_policy_postgres.py`](44_policy_postgres.py) | Shared approvals, exactly-once effects, and reconciliation on PostgreSQL. |

## Reference applications

Each application is an independent FastAPI and React project with its own
`pyproject.toml`, `uv.lock`, tests, container files, and README. Install one
with `uv sync --locked` in its directory.

| Application | What it does |
|---|---|
| [Enterprise Knowledge Assistant](enterprise_knowledge_assistant/) | Grounded, cited answers over a set of enterprise documents. |
| [QA Customer Chatbot](qa_customer_chatbot/) | A customer-support agent with operational tools, feedback, and BehaviorWeave guardrails. |
| [RCA Generator](rca_generator/) | Structured root-cause analyses from incident events, logs, and metrics. |
| [Web Compliance Auditor](web_compliance_auditor/) | Audits a public URL against policies with MCP browser evidence, pgvector retrieval, and governed publication. |
| [Engineering Change Investigator](engineering_change_investigator/) | Go/no-go release decisions from engineering records and live evidence, published under approval. |
| [Autonomous Product Studio](autonomous_product_studio/) | Builds and validates a product in a private workspace with Loop Engineering and exactly-once publishing. |
| [Continuous Evidence Lab](continuous_evidence_lab/) | Keeps cited evidence current while it changes, with refresh, replanning, and human review. |

Shared pieces used by the applications:

- [`observability/`](observability/): Prometheus, Grafana, and Langfuse for
  local monitoring.
- [`keycloak/`](keycloak/): the identity provider bootstrap for the
  applications that sign users in.
