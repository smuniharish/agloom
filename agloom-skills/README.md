# Agloom Agent Skill

This directory holds the [Agent Skill](https://agentskills.io/specification)
for Agloom. It teaches coding agents to build, configure, debug, and test
applications that use the `agloom` package.

| Path | Contents |
|---|---|
| [`skills/agloom/SKILL.md`](skills/agloom/SKILL.md) | Instructions loaded when the skill activates. |
| [`skills/agloom/references/`](skills/agloom/references/) | Focused references loaded on demand: API, topologies, Policy Manager, Loop Engineering, integrations, and troubleshooting. |
| [`validation/`](validation/) | How the skill is checked before a release. |

## Install

Copy `skills/agloom/` into the skills directory of your agent, for example
`.github/skills/agloom/` or `.claude/skills/agloom/` in a repository. See
[Agent Skills](https://agloom.readthedocs.io/en/latest/agent-skills/) in the
documentation for details.

## Maintain

The skill describes the released package. When the public API or documented
behavior changes:

1. Update `SKILL.md` and only the affected references.
2. Check every claim against the package source, tests, examples, or
   documentation.
3. Run the checks in [`validation/README.md`](validation/README.md).

The skill is covered by the repository's Apache License 2.0.
