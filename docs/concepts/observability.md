# Observability

Agloom emits normalized runtime events and propagates LangChain callbacks from
both capability-aware analysis and execution. `create_agent(observers=[...])`
can fan out to several observer implementations:

- `LoggingObserver` emits structured events through `structlog`.
- `PrometheusObserver` exports lifecycle counters, active executions, and
  duration histograms.
- `LangfuseObserver` records Agloom events and supplies the Langfuse LangChain
  callback handler for model, chain, retriever, and tool observations.

```python
import os

from agloom import LangfuseObserver, PrometheusObserver, create_agent

metrics = PrometheusObserver()
metrics.start_http_server(port=9464)
langfuse = LangfuseObserver(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)
agent = create_agent(model=model, tools=[], observers=[metrics, langfuse])

try:
    agent.invoke("Say hello.")
finally:
    agent.close_observability()
```

## Explainability by default

Every agent automatically owns an isolated `langgraph-xai` `XAIRuntime`.
Agloom instruments capability-aware analysis, DIRECT model execution,
streaming, and every compiled topology. The runtime captures canonical
executions, state transitions, model/tool callbacks, provenance, attribution,
and deterministic structured explanations using langgraph-xai's in-memory
defaults.

Use `agent.xai` to access the runtime, `xai_options` to override its constructor
defaults, or `explainability` to inject a fully configured runtime. When a
`LangfuseObserver` is present, its client is automatically registered as
langgraph-xai's canonical-event observability provider. Set
`xai_enabled=False` only when explainability must be explicitly disabled.

## Local stack

The `observability/compose.yaml` stack provisions:

- Prometheus at `http://localhost:9090`;
- Grafana at `http://localhost:3001`, with a provisioned Prometheus datasource
  and Agloom dashboard;
- Langfuse at `http://localhost:3000`;
- the PostgreSQL, ClickHouse, Redis, and MinIO services required by Langfuse.

Create a private `.env` from `.env.example`, replace every placeholder, then
run:

```console
docker compose --env-file .env -f observability/compose.yaml up -d
```

The Agloom process must expose Prometheus on port `9464`.
Prometheus reaches that host endpoint as `host.docker.internal:9464`.
All public service bindings are restricted to loopback.
