from _support import real_model
from pydantic import BaseModel

from agloom import create_agent
from agloom.capabilities.integrations import with_structured_output


class Summary(BaseModel):
    title: str
    complete: bool


agent = create_agent(
    model=real_model(),
    tools=[],
)
structured = with_structured_output(agent, Summary)
result = structured.invoke(
    "Return a Summary whose title is exactly 'Quarterly readiness report' and "
    "whose complete field is true."
).structured
assert result == Summary(title="Quarterly readiness report", complete=True)
print(result)
