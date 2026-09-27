# Observability API

Pass one or more observers through `create_agent(observers=[...])`.

## Built-in observers

### `LoggingObserver`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `logger` | Structured logger or `None` | Optional | `None` | Logger with an `info(event, **fields)` method. Uses `structlog.get_logger("agloom.events")` when omitted. |

### `PrometheusObserver`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `registry` | `CollectorRegistry` | Optional | Prometheus `REGISTRY` | Registry that receives Agloom metrics. |
| `namespace` | `str` | Optional | `"agloom"` | Prefix for emitted metric names. |

`start_http_server(port=9464, *, address="127.0.0.1")` starts the metrics
endpoint. `port` is optional and defaults to `9464`; `address` is optional and
defaults to loopback.

### `LangfuseObserver`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `client` | `Langfuse \| None` | Optional | `None` | Existing Langfuse client. Cannot be combined with client options. |
| `callback_handler` | `CallbackHandler \| None` | Optional | `None` | Existing LangChain callback handler. |
| `event_metadata` | `Mapping[str, Any] \| None` | Optional | `None` | Metadata attached to every Agloom event. |
| `**client_options` | `Any` | Optional | `{}` | Options forwarded to `Langfuse` when `client` is omitted. |

### `CallableObserver`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `sink` | `Callable[[AgentEvent], None]` | **Required** | — | Callback invoked for every normalized runtime event. |

### `CompositeObserver`

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `observers` | `Iterable[Observer]` | Optional | `()` | Observers invoked in order; shutdown runs in reverse order. |

## Custom observers

Subclass `Observer` and implement `on_event(event)`. Override
`callback_handlers`, `flush()`, or `shutdown()` only when needed. Observer
errors propagate so telemetry failures are visible.

## Environment-driven helper

`application_observers(application_id, *, default_metrics_port)` creates
logging and Prometheus observers, plus Langfuse when both Langfuse keys are
configured.

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `application_id` | `str` | **Required** | — | Stable application identifier used in metric names and event metadata. |
| `default_metrics_port` | `int` | **Required** | — | Port used unless `AGLOOM_METRICS_PORT` is set. |

`observability_enabled()` reads `AGLOOM_OBSERVABILITY_ENABLED`.
`observability_status(default_metrics_port)` returns non-sensitive effective
telemetry settings.

See the [observability guide](../concepts/observability.md) for a complete
setup.
