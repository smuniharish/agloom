from _support import real_model, verified_text

from agloom import WorkerSpec, create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="blackboard",
    workers=[
        WorkerSpec(
            name="latency_analyst",
            description="Analyze only the stated latency measurements.",
            instructions="Record that option A is 120 ms and option B is 200 ms.",
        ),
        WorkerSpec(
            name="cost_analyst",
            description="Analyze only the stated per-request costs.",
            instructions="Record that option A costs $0.02 and option B costs $0.01.",
        ),
    ],
)
result = agent.invoke(
    "Compare two API options. Given facts: A has 120 ms latency and costs "
    "$0.02/request; B has 200 ms latency and costs $0.01/request. Synthesize "
    "the latency and cost trade-off without inventing facts."
)
print(verified_text(result, "120", "200", "$0.02", "$0.01"))
