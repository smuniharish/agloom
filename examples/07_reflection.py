from _support import real_model, verified_text

from agloom import create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="reflection",
)
result = agent.invoke(
    "Write one accurate sentence explaining that Earth takes about 365.25 days "
    "to orbit the Sun. Preserve the number 365.25 during review and revision."
)
print(verified_text(result, "365.25", "Sun"))
