from _support import real_model, verified_text

from agloom import WorkerSpec, create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="supervisor",
    workers=[
        WorkerSpec(
            name="calculator",
            description="Solve deterministic arithmetic word problems.",
            instructions=(
                "Calculate the requested result and answer in one sentence. "
                "Do not ask for more context."
            ),
        ),
        WorkerSpec(
            name="writer",
            description="Edit prose without performing calculations.",
        ),
    ],
)
result = agent.invoke(
    "A project budget is $1,200 and $450 has been spent. State the remaining "
    "budget in one sentence; the result must include $750."
)
print(verified_text(result, "750"))
