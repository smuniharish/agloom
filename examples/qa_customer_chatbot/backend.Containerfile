FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY examples/qa_customer_chatbot /workspace/examples/qa_customer_chatbot

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"
RUN uv sync \
    --project examples/qa_customer_chatbot \
    --frozen \
    --no-dev

WORKDIR /workspace/examples/qa_customer_chatbot
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
