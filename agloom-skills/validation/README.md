# Validating the Agloom skill

## Automated checks

Continuous integration runs the reference validator from the Agent Skills
project on every change:

```bash
uvx --from skills-ref==0.1.1 agentskills validate agloom-skills/skills/agloom
```

It checks the frontmatter: `name` matches the directory, `description` is
present and at most 1,024 characters, and only specified fields are used.

## Review checklist

1. `SKILL.md` stays under 500 lines, and references are linked directly from
   it (one level deep).
2. Every relative link resolves.
3. Every import, argument, method, event, and error name exists in the
   released package (`agloom.__all__` and the `create_agent` signature).
4. Behavior claims match the tests, the runnable examples, or the
   [documentation](https://agloom.readthedocs.io/en/latest/).
5. No private modules, internal configuration keys, credentials, or
   superseded APIs appear.

## Activation scenarios

| Request | Activates | Expected guidance |
|---|---|---|
| "Let the agent choose how to run each task." | Yes | Omit `pattern`; automatic selection. |
| "Always use the Planner for this workflow." | Yes | `pattern="planner"`, no analysis call. |
| "Only allow Planner or Supervisor." | Yes | An allow-list, not a composition. |
| "Run ReAct, then Reflection." | Yes | `pattern="hybrid"` with `composition`. |
| "Require approval for refunds over 100." | Yes | `PolicyManager` rule with `REQUIRE_APPROVAL`, a consequential tool, and a checkpointer. |
| "Load our own policy file per tenant." | Yes | `PolicyManager(policy=path)` with a versioned JSON policy. |
| "Iterate until the report is complete." | Yes | `LoopPolicy(max_iterations=...)` with an evaluator. |
| "Let a reviewer edit a tool call before it runs." | Yes | LangChain `HumanInTheLoopMiddleware` and `resume`. |
| "Add MCP tools from our server." | Yes | `MCPRuntime` with `RefreshPolicy(on_register=True)` and async APIs. |
| "Bound recursive subtasks." | Yes | `ExecutionLimits(max_depth=..., max_children=...)`. |
| "Debug a plain LangGraph graph that doesn't use Agloom." | No | General LangGraph guidance. |
