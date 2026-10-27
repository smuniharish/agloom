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
COPY examples/qa_customer_chatbot/pyproject.toml examples/qa_customer_chatbot/uv.lock /workspace/examples/qa_customer_chatbot/
RUN uv sync --project examples/qa_customer_chatbot --frozen --no-dev

COPY examples/qa_customer_chatbot/backend /workspace/examples/qa_customer_chatbot/backend
COPY examples/qa_customer_chatbot/data /workspace/examples/qa_customer_chatbot/data

WORKDIR /workspace/examples/qa_customer_chatbot
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
EXPOSE 8000 9465
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
