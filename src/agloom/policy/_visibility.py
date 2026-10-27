"""Static analysis of agentbarrier policies."""

from __future__ import annotations

from fnmatch import fnmatchcase

from agentbarrier.runtime import PolicyEffect, RuntimePolicy


def certainly_denied(policy: RuntimePolicy, tool_name: str) -> bool:
    """Return whether the policy denies every possible call of a tool.

    Mirrors `RuntimePolicy.evaluate`: rules are checked in order and the first
    matching rule wins. A conditional deny rule only denies some calls, so the
    analysis continues past it; any rule that could allow or require approval
    makes the tool reachable. The result is sound: when it is `True`,
    `policy.evaluate(tool_name, arguments)` is `deny` for all arguments.
    """
    for rule in policy.rules:
        if not fnmatchcase(tool_name, rule.tool):
            continue
        if rule.effect is not PolicyEffect.DENY:
            return False
        if not rule.conditions:
            return True
    return policy.default_effect is PolicyEffect.DENY
