# Playwright's image provides Chromium, its system libraries, and Node.js for
# the stdio MCP servers; uv installs Python 3.14 and the locked dependencies.
FROM mcr.microsoft.com/playwright:v1.63.0-noble
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.14 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MCP_SERVER_LAUNCHER=installed \
    BROWSER_EXECUTABLE=/usr/local/bin/chromium \
    BROWSER_SANDBOX=false

# Keep these versions in sync with backend/mcp_servers.py.
RUN npm install --global \
       @playwright/mcp@0.0.83 \
       @modelcontextprotocol/server-filesystem@2026.8.31 \
       @modelcontextprotocol/server-everything@2026.8.31 \
    && ln -s "$(find /ms-playwright -type f -path '*/chrome-linux*/chrome' | head -n 1)" \
       /usr/local/bin/chromium

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples/web_compliance_auditor/pyproject.toml \
     examples/web_compliance_auditor/uv.lock \
     examples/web_compliance_auditor/
RUN uv sync --project examples/web_compliance_auditor --frozen --no-dev
COPY examples/web_compliance_auditor examples/web_compliance_auditor

WORKDIR /workspace/examples/web_compliance_auditor
EXPOSE 8003 9467
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8003"]
