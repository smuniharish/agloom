FROM node:24-bookworm-slim AS node

FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY examples/continuous_evidence_lab/pyproject.toml \
     examples/continuous_evidence_lab/uv.lock \
     /workspace/examples/continuous_evidence_lab/
RUN uv sync --project examples/continuous_evidence_lab --frozen --no-dev --no-editable

FROM python:3.14-slim
COPY --from=node /usr/local /usr/local
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN npm install --global @modelcontextprotocol/server-filesystem@2026.8.31 \
    @modelcontextprotocol/server-everything@2026.8.31 \
 && npm cache clean --force \
 && useradd --create-home --uid 10001 evidence
WORKDIR /app
COPY --chown=evidence:evidence examples/continuous_evidence_lab /app
USER evidence
EXPOSE 8007 9470
CMD ["uvicorn", "backend.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8007"]