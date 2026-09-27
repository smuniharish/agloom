from _support import real_model, verified_text
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agloom import create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="pipeline",
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=lambda thread_id: thread_id == "checkpoint-example",
    interrupt_before=["stage_0_analyze"],
)
config = {"configurable": {"thread_id": "checkpoint-example"}}
task = (
    "Write one status sentence stating that the API migration is complete, "
    "48 tests passed, and deployment is Friday."
)
agent.invoke(task, config=config)
result = agent.invoke(Command(resume=True), config=config)
print(agent.get_state(config).values["output"])
print(verified_text(result, "complete", "48", "Friday"))
