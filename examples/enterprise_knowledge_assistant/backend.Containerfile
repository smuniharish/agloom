FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY examples/enterprise_knowledge_assistant/pyproject.toml examples/enterprise_knowledge_assistant/uv.lock /workspace/examples/enterprise_knowledge_assistant/
RUN uv sync --project examples/enterprise_knowledge_assistant --frozen --no-dev

COPY examples/enterprise_knowledge_assistant/backend /workspace/examples/enterprise_knowledge_assistant/backend
COPY examples/enterprise_knowledge_assistant/data /workspace/examples/enterprise_knowledge_assistant/data

WORKDIR /workspace/examples/enterprise_knowledge_assistant
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
EXPOSE 8000 9464
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
