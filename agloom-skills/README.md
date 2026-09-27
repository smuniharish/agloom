# Agloom Agent Skills

This directory is the canonical Agent Skills distribution for Agloom. It
contains procedural guidance for coding agents that need to integrate,
configure, test, debug, or extend the existing `agloom` package.

It is not a Python package, an execution topology, a runtime capability
catalog, or a replacement for Agloom's `create_agent()` API.

| Component | Location | Purpose |
| --- | --- | --- |
| Agloom runtime | [`src/agloom/`](https://github.com/smuniharish/agloom/tree/master/src/agloom) | Published Python package and supported public API. |
| Agloom Agent Skill | [`skills/agloom/`](skills/agloom/) | Canonical agent-oriented instructions and focused references. |
| Skill validation | [`validation/`](validation/) | Structural checks, source-accuracy rules, and activation matrix. |

## Agent Skills format

The canonical skill follows the
[Agent Skills specification](https://agentskills.io/specification): a named
directory containing `SKILL.md` with required `name` and `description` YAML
frontmatter. The name matches the containing directory (`agloom`), and no
host-specific copies are maintained.

Compatible agents should load
[`skills/agloom/SKILL.md`](skills/agloom/SKILL.md) for work involving Agloom
construction, execution selection, topologies, capabilities, recursion,
checkpointing, structured output, or observability.

Installation instructions are in
[Agent Skills - Agloom](https://agloom.readthedocs.io/en/latest/agent-skills/).

## Maintaining the distribution

When Agloom's public API or documented behavior changes:

1. Update the canonical skill and only the affected references.
2. Verify every claim against the public package API, tests,
   [examples](https://github.com/smuniharish/agloom/tree/master/examples), or
   [documentation](https://agloom.readthedocs.io/en/latest/).
3. Follow [`validation/README.md`](validation/README.md).
4. Do not create separate Claude, Codex, Cursor, or Copilot copies.

This distribution is covered by the repository's Apache License 2.0.
