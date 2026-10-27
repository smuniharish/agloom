# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |
| Earlier releases | No |

Security fixes are released as patch versions of the latest supported line.
Upgrade to the newest patch release before reporting an issue.

## Reporting a vulnerability

Report vulnerabilities privately with GitHub's **Report a vulnerability** form
in the repository's **Security** tab. This creates a private advisory visible
only to the maintainers. Do not disclose a suspected vulnerability in a public
issue, discussion, or pull request.

Include the affected version, the impact, reproduction steps or a proof of
concept, and any known mitigations. Remove credentials, personal data, and
unrelated proprietary information from the report.

Maintainers acknowledge reports within five business days, investigate and
coordinate a fix privately, and credit reporters who want attribution.
Timelines depend on severity and complexity. Please wait for a fixed release,
or a coordinated date, before disclosing publicly.

## Security model

Agloom runs inside your application and treats model output as untrusted. Its
controls are enforced in code, not requested in prompts.

**Policy enforcement.** A `PolicyManager` checks every tool call against an
agentbarrier policy before the tool runs, including tools contributed by
middleware and calls your code makes with `execute_capability`. Arguments are
validated against the tool's schema first, so rules judge exactly the values
the tool receives, and invalid calls never run. Rules are ordered, the first
match wins, and calls that no rule allows are denied. Tools that the policy
denies for every call are never offered to the model. Policy enforcement is
the innermost layer around each tool, so every call that reaches the tool,
including retries by other middleware, passes through it. Capability routing
additionally refuses calls to tools that were not offered for the request, but
it is not an authorization mechanism; the policy is.

**Approvals.** Rules can require a human decision. Decisions are made in
agentbarrier (API, CLI, dashboard, or Slack), not by the agent, and are bound
to the exact tool call and arguments. Approvals can expire, and a rejected or
expired request fails closed.

**Exactly-once side effects.** Tools declared consequential are claimed with a
business idempotency key: retries, replays, and resumes return the recorded
result instead of executing again, and reusing a key for a different request
is rejected. Side effects fail closed: when a consequential tool raises, or the
process dies while it runs, the action's outcome is recorded as unknown and
the action is held for operator reconciliation in agentbarrier rather than
repeated; a checkpointed run pauses until the action is reconciled. Every
action and every reconciliation leaves a hash-chained audit receipt.

**Checkpoint authorization.** A checkpointer cannot be configured without a
`checkpoint_authorizer`. Every invocation, stream, resume, and state read must
name a `thread_id` that the authorizer accepts, so one caller cannot read or
resume another caller's thread. Checkpoints contain only JSON-compatible values
and LangChain messages and work with LangGraph's strict serialization.

**Prompt-injection hygiene.** Trusted instructions come only from
`system_prompt` and your topology configuration. Input messages may contain
only user and assistant turns, so callers cannot inject system instructions or
forge tool results. Task-analysis notes, Loop Engineering context (evaluator
feedback, previous results, errors), and human feedback are passed to the
model as labelled data in the user turn, never as system instructions. Task
analysis treats the request and the capability inventory as untrusted data.
Analysis can never grant a capability: requested actions that no available
tool can perform are passed to the model as actions it must not claim to have
performed.

**Bounded execution.** Recursion depth, child executions, parallel fan-out,
plan length, handoffs, wall-clock time, loop iterations, active duration, and
tool calls are bounded by validated limits, and agents can be cancelled
cooperatively.

**Secrets.** Agloom never reads or stores credentials; model clients, stores,
and observers obtain them from your environment or secret manager. Runtime
event payloads never contain prompts, tool arguments, or credentials. Traces
recorded by callback-based observers, such as Langfuse, can contain prompts and
tool arguments; apply your data-handling rules to them.

**Your responsibilities.** Authenticate users and authorize thread ownership
in your `checkpoint_authorizer`; give every tool the narrowest policy it needs;
declare side-effecting tools consequential; isolate code execution and file
access in sandboxes; treat tool results, retrieved content, and MCP servers as
untrusted; and keep dependencies up to date.

## Release integrity

Release artifacts are built in GitHub Actions, accompanied by a CycloneDX
SBOM, and attested with GitHub artifact provenance. PyPI publication uses OIDC
trusted publishing; no long-lived publishing credential is stored.
