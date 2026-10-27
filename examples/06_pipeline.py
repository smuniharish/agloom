"""Pipeline: fixed, ordered stages that each build on the previous output.

Every stage is a tool-capable agent with its own instruction; an optional
`transform` post-processes a stage's text deterministically. Use a pipeline
when the work always passes through the same steps, such as extract,
compute, and format.

Run:
    uv run python examples/06_pipeline.py
"""

from _support import real_model, verified_text

from agloom import PipelineStage, create_agent


def as_report(text: str) -> str:
    """Normalize the final stage's text into a one-paragraph report."""
    return "Sensor inventory: " + " ".join(text.replace("*", "").split())


def main() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="pipeline",
        stages=[
            PipelineStage(
                name="extract",
                instruction="List the quantities stated in the request.",
            ),
            PipelineStage(
                name="compute",
                instruction="Compute the total and show the arithmetic.",
            ),
            PipelineStage(
                name="summarize",
                instruction="State the final total in one plain sentence.",
                transform=as_report,
            ),
        ],
    )

    answer = agent.invoke(
        "A warehouse has 3 boxes with 8 sensors in each box. How many sensors "
        "does it hold in total?"
    )

    text = verified_text(answer, "24")
    assert text.startswith("Sensor inventory: "), text
    print(text)


if __name__ == "__main__":
    main()
