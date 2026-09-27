from _support import real_model, verified_text

from agloom import WorkerSpec, create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="swarm",
    workers=[
        WorkerSpec(
            name="writer",
            description="Draft the requested conversion explanation.",
            instructions=(
                "Draft one sentence showing that 392°F equals 200°C, then route "
                "to reviewer with NEXT: reviewer."
            ),
        ),
        WorkerSpec(
            name="reviewer",
            description="Verify conversions and finalize the response.",
            instructions=(
                "Verify the prior conversion, return a polished sentence that "
                "includes 392°F and 200°C, then end with NEXT: STOP."
            ),
        ),
    ],
)
result = agent.invoke(
    "Convert 392 degrees Fahrenheit to Celsius and provide the verified result."
)
print(verified_text(result, "392", "200", "NEXT: STOP"))
