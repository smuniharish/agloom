from _support import real_model, verified_text

from agloom import RecursionPolicy, create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(max_depth=2, max_subtasks=8),
)
result = agent.invoke(
    "Create exactly two self-contained subtasks, one per line with no "
    "sub-bullets: first state that Agloom has eight execution topologies; "
    "second state that DIRECT is an execution mode, not a topology. Treat "
    "these supplied facts as authoritative; restate them without research or "
    "external verification."
)
print(verified_text(result, "eight", "DIRECT", "not a topology"))
