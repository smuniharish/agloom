# Agent Skills

Agloom publishes an [Agent Skill](https://agentskills.io/) that teaches coding
agents how to build with Agloom: choosing execution modes and topologies,
configuring capabilities and MCP, writing policies, setting up Loop
Engineering, checkpointing, streaming, observability, and testing. The skill
is guidance for your coding agent; it adds nothing to your application at run
time.

The canonical skill lives in the repository at
[`agloom-skills/skills/agloom`](https://github.com/smuniharish/agloom/tree/master/agloom-skills/skills/agloom).
It follows the [Agent Skills specification](https://agentskills.io/specification):
a `SKILL.md` with `name` and `description` frontmatter, plus reference files
that the agent reads when a task needs them. One copy serves every compatible
host.

## Install with the skills CLI

```bash
npx skills add https://github.com/smuniharish/agloom/tree/master/agloom-skills/skills/agloom
```

Follow the CLI's prompts to choose the target agents and scope. Make sure the
whole `agloom` directory is installed, not only `SKILL.md`.

## Install manually

Copy the complete `agloom-skills/skills/agloom` directory, keeping its name, to
the skills location of your coding agent:

| Coding agent | This repository only | All your projects |
|---|---|---|
| Claude Code | `.claude/skills/agloom/` | `~/.claude/skills/agloom/` |
| Codex | `.agents/skills/agloom/` | `~/.agents/skills/agloom/` |
| Cursor | `.agents/skills/agloom/` or `.cursor/skills/agloom/` | `~/.agents/skills/agloom/` or `~/.cursor/skills/agloom/` |
| GitHub Copilot | `.agents/skills/agloom/`, `.github/skills/agloom/`, or `.claude/skills/agloom/` | `~/.agents/skills/agloom/` or `~/.copilot/skills/agloom/` |

For other Agent Skills-compatible hosts, copy the directory to the location
their documentation names. Reload or restart the agent after installing.

## Use the skill

Coding agents activate the skill from its description when a task involves
Agloom, for example "add a refund approval policy to this Agloom agent" or
"make this agent iterate until the report passes validation". Most hosts also
let you invoke it explicitly, typically as `/agloom` (or `$agloom` in Codex).

## Keep it current

The skill is versioned with the library. When you upgrade Agloom, update the
installed skill too: rerun the CLI, or replace the copied directory with the
version from the matching release.
