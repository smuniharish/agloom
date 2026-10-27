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
COPY examples/rca_generator/pyproject.toml examples/rca_generator/uv.lock /workspace/examples/rca_generator/
RUN uv sync --project examples/rca_generator --frozen --no-dev

COPY examples/rca_generator/backend /workspace/examples/rca_generator/backend

WORKDIR /workspace/examples/rca_generator
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
EXPOSE 8000 9466
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
