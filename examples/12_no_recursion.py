from _support import real_model, verified_text

from agloom import create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="planner",
    recursion=False,
)
result = agent.invoke(
    "Create one self-contained subtask that states this key point exactly: "
    "DIRECT is not one of Agloom's eight topologies."
)
print(verified_text(result, "DIRECT", "eight", "not"))
