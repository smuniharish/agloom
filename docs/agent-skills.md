# Agent Skills

Agloom publishes one portable Agent Skill that teaches coding agents how to
integrate, configure, debug, test, and extend the existing Agloom framework.
The skill is procedural guidance, not a Python runtime component, execution
topology, or replacement for `create_agent()`.

The
[canonical Agent Skill](https://github.com/smuniharish/agloom/tree/master/agloom-skills/skills/agloom)
is portable across compatible coding agents. Agloom maintains one authoritative
copy instead of separate instructions for every host.

The skill follows the
[Agent Skills specification](https://agentskills.io/specification) and uses
only the required `name` and `description` frontmatter. Agloom does not
maintain separate Claude, Codex, Cursor, or Copilot copies.

## What the skill teaches

The `agloom` skill activates for coding tasks involving:

- automatic, constrained, or explicit execution selection;
- DIRECT and the eight execution topologies;
- explicit Hybrid composition;
- LangChain tools, application capabilities, and MCP capabilities;
- policy, embeddings, retrieval, and reranking;
- bounded recursion;
- checkpointing, human approval, resume, streaming, and cancellation;
- feedback, behavior, context, refresh, explainability, and structured output;
- Prometheus, Grafana, Langfuse, and lifecycle observability;
- debugging and testing an Agloom integration.

It also teaches critical boundaries: DIRECT is not a topology, a pattern list
is not Hybrid composition, orchestration scaling is not infrastructure
autoscaling, and application-owned credentials, persistence, authentication,
and deployment stay outside Agloom.

## Install from skills.sh

Install the canonical skill directory directly:

```console
npx skills add https://github.com/smuniharish/agloom/tree/master/agloom-skills/skills/agloom
```

Follow the CLI's current target-selection prompts. Verify that it installs the
complete `agloom` directory, including `SKILL.md` and `references/`.

## Install manually

Copy the complete canonical directory:

```text
agloom/
├── SKILL.md
└── references/
    ├── architecture.md
    ├── configuration.md
    ├── integration.md
    └── troubleshooting.md
```

Do not copy only `SKILL.md`; it links to the bundled references.

### Claude Code

| Scope | Destination |
| --- | --- |
| Current repository | `.claude/skills/agloom/` |
| All local projects | `~/.claude/skills/agloom/` |

Restart or reload Claude Code after copying. Claude can activate the skill from
its description or explicitly with `/agloom`.

### Codex

| Scope | Destination |
| --- | --- |
| Current repository | `.agents/skills/agloom/` |
| All local projects | `~/.agents/skills/agloom/` |

Codex can activate the skill from its description or explicitly with
`$agloom`. Use `/skills` to inspect available skills.

### Cursor

| Scope | Destination |
| --- | --- |
| Current repository | `.agents/skills/agloom/` or `.cursor/skills/agloom/` |
| All local projects | `~/.agents/skills/agloom/` or `~/.cursor/skills/agloom/` |

Restart Cursor after copying. In Agent chat, type `/` and select `agloom`, or
allow description-based activation.

### GitHub Copilot

| Scope | Destination |
| --- | --- |
| Current repository | `.agents/skills/agloom/`, `.github/skills/agloom/`, or `.claude/skills/agloom/` |
| All local projects | `~/.agents/skills/agloom/` or `~/.copilot/skills/agloom/` |

For Copilot CLI, start a new session or run `/skills reload`. Verify discovery
with `/skills info agloom`; invoke explicitly with `/agloom`.

### Other Agent Skills-compatible hosts

Copy the complete `agloom` directory to the host's documented skills location.
If a host requires metadata or a plugin manifest, add only a thin adapter that
points to this canonical skill. Do not duplicate the instructions.

## Update and verify

For a manual installation, replace the complete installed `agloom` directory
with the latest repository version, then restart or reload the host.

Verify all of the following:

1. The directory is named `agloom`.
2. `SKILL.md` and all four files under `references/` are present.
3. The host lists `agloom` as available, if it exposes a skill listing.
4. A task such as “add bounded recursion to this Agloom planner” activates or
   can explicitly invoke the skill.
5. An unrelated raw LangGraph or infrastructure-autoscaling task does not
   activate the skill merely because it mentions agents.
