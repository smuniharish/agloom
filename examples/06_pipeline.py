from _support import real_model, verified_text

from agloom import PipelineStage, create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="pipeline",
    stages=[
        PipelineStage(
            name="calculate",
            instruction="Calculate the answer and show the arithmetic.",
        ),
        PipelineStage(
            name="explain",
            instruction="Explain the prior calculation in one concise sentence.",
        ),
        PipelineStage(
            name="verify",
            instruction=(
                "Verify the prior result and return the final answer. Preserve "
                "the exact numeric result."
            ),
        ),
    ],
)
result = agent.invoke(
    "There are 3 boxes with 8 sensors in each box. How many sensors are there? "
    "The final answer must include 24."
)
print(verified_text(result, "24"))
