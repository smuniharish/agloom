from _support import real_model, verified_text

from agloom import create_agent

agent = create_agent(
    model=real_model(),
    tools=[],
    pattern="planner",
)
result = agent.invoke(
    "Plan a three-box move using only these facts: box A weighs 2 kg, box B "
    "weighs 3 kg, box C weighs 5 kg, and each trip can carry at most 6 kg. "
    "Create exactly three self-contained subtasks: calculate the 10 kg total, "
    "calculate that at least 2 trips are needed, and label the boxes A, B, C. "
    "Use one numbered subtask per line with no sub-bullets."
)
print(
    verified_text(
        result,
        "10",
        "2",
        any_of=("A", "box A"),
    )
)
