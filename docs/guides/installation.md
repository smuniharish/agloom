# Installation

Agloom supports Python 3.12. Install it with uv:

```console
uv add agloom
```

The equivalent pip command is:

```console
python -m pip install agloom
```

LangChain, LangGraph, and all seven supported capability packages are installed
with Agloom. Add the LangChain provider package used by your application, for
example:

```console
uv add langchain-openai
```

Agloom core does not require LangSmith credentials, a database, containers, or
hosted infrastructure.

For repository development:

```console
uv sync --frozen --group dev
```
