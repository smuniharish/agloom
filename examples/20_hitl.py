from _support import real_model, verified_text
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agloom import create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="pipeline",
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=lambda thread_id: thread_id == "approval-example",
    interrupt_before=["stage_0_analyze"],
)
config = {"configurable": {"thread_id": "approval-example"}}
task = (
    "Write one status sentence stating that the API migration is complete, "
    "48 tests passed, and deployment is Friday."
)
paused = agent.invoke(task, config=config)
print("Approval required:", bool(paused.get("__interrupt__")))
result = agent.invoke(Command(resume=True), config=config)
print(verified_text(result, "complete", "48", "Friday"))
