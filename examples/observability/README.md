# Observability stack

A local Prometheus, Grafana, and Langfuse stack for the Agloom reference
applications. Prometheus scrapes each application's `PrometheusObserver`
endpoint, Grafana shows the provisioned **Agloom Runtime** dashboard, and
Langfuse receives traces from `LangfuseObserver`.

| Service | URL |
|---|---|
| Grafana | <http://localhost:3001> |
| Prometheus | <http://localhost:9090> |
| Langfuse | <http://localhost:3000> |

## Start the stack

1. Copy the repository's `.env.example` to `.env` and replace every
   `replace-with-...` value in the "Self-hosted observability stack" section
   with a unique secret. `LANGFUSE_ENCRYPTION_KEY` must be 64 hexadecimal
   characters (`openssl rand -hex 32`).
2. Start the stack from the repository root:

   ```bash
   docker compose --env-file .env -f examples/observability/compose.yaml up -d
   ```

3. Sign in to Langfuse with `LANGFUSE_INIT_USER_EMAIL` and
   `LANGFUSE_INIT_USER_PASSWORD`. The project keys `LANGFUSE_PUBLIC_KEY` and
   `LANGFUSE_SECRET_KEY` are provisioned on first start.

The `apps` profile also builds and runs the Enterprise Knowledge Assistant,
QA Customer Chatbot, and RCA Generator backends with metrics and tracing
enabled:

```bash
docker compose --env-file .env -f examples/observability/compose.yaml --profile apps up -d --build
```

The Web Compliance Auditor, Engineering Change Investigator, Autonomous Product
Studio, and Continuous Evidence Lab join this stack's
`agloom-observability_default` network from their own Compose files, so start
this stack first.

## Metrics and alerts

Every application exports the metrics documented for
[`PrometheusObserver`](https://agloom.readthedocs.io/en/latest/concepts/observability/)
under its own `agloom_<application>` prefix. The dashboard groups them into
runtime, Policy Manager, and Loop Engineering panels and can be filtered by
application.

`prometheus/alerts.yml` defines these alerts:

| Alert | Fires when |
|---|---|
| `AgloomMetricsTargetDown` | An application cannot be scraped for 2 minutes. |
| `AgloomExecutionFailureRatioHigh` | More than 5% of at least 20 executions fail within 10 minutes. |
| `AgloomConsequentialActionFailed` | A consequential action failed. Its outcome is unknown, so reconcile it in the agentbarrier store before it can run again. |
| `AgloomPolicyDenialsHigh` | The agent keeps requesting a tool that the policy denies. |
| `AgloomLoopsStopWithoutObjective` | More than 25% of loops stop on budgets, stagnation, or oscillation. |

Langfuse's ClickHouse, Redis, PostgreSQL, and MinIO services use the versions
that Langfuse pins in its own Compose file.
