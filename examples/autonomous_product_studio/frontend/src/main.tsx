import React, { FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import keycloak, { authenticatedFetch, initializeAuth, streamSse } from "./auth";

type Criterion = { id: string; description: string; kind: string; target: string; expected?: string };
type Project = { id: string; name: string; objective: string; acceptance_criteria: Criterion[] };
type Task = { id: string; title: string; status: string };
type RunStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "awaiting_reconciliation"
  | "completed"
  | "failed"
  | "cancelled"
  | "rejected";
type Approval = {
  action_id: string;
  tool: string;
  kind: "approval" | "reconciliation";
  arguments: Record<string, unknown>;
  policy_rule: string;
  error?: string | null;
  status: string;
  decided_by?: string | null;
  decision_reason?: string | null;
};
type Run = {
  id: string;
  status: RunStatus;
  approvals: Approval[];
  proof: {
    iterations: number;
    acceptance_criteria: number;
    passed_criteria: number;
    validation_pass_rate: number;
    tool_calls: number;
    model_calls: number;
    elapsed_seconds: number;
    termination_reason?: string;
  };
};
type RunEvent = {
  id?: number;
  type: string;
  timestamp: string;
  message?: string;
  name?: string;
  iteration?: number;
  progress?: number;
  action?: string;
  run?: Run;
};

const ACTIVE: RunStatus[] = ["queued", "running", "awaiting_approval", "awaiting_reconciliation"];
const WAITING: RunStatus[] = ["awaiting_approval", "awaiting_reconciliation"];
const TERMINAL_EVENTS = ["completed", "failed", "cancelled", "rejected"];
const HEADLINES: Record<RunStatus, string> = {
  queued: "Waiting for a worker",
  running: "Loop in progress",
  awaiting_approval: "Release awaiting review",
  awaiting_reconciliation: "Release outcome unknown",
  completed: "Objective validated",
  failed: "Budget ended",
  cancelled: "Run cancelled",
  rejected: "Release rejected",
};

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ReviewPanel({ run, onDecided }: { run: Run; onDecided: (run: Run) => void }) {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    authenticatedFetch(`/api/runs/${run.id}/approvals`)
      .then((response) => response.json())
      .then(setApprovals)
      .catch(() => setError("Pending approvals are unavailable."));
  }, [run.id, run.status]);

  async function submit(path: string, body: Record<string, unknown>) {
    setBusy(true);
    setError("");
    try {
      const response = await authenticatedFetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await response.text());
      onDecided(await response.json());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The decision failed");
    } finally {
      setBusy(false);
    }
  }

  function decide(approval: Approval, decision: "approve" | "reject") {
    return submit(`/api/runs/${run.id}/approvals/${approval.action_id}`, { decision, reason: reason || null });
  }

  function reconcile(approval: Approval, outcome: "committed" | "not_committed") {
    return submit(`/api/runs/${run.id}/reconciliations/${approval.action_id}`, { outcome, reason });
  }

  const pending = approvals.filter((approval) => approval.kind === "approval" && approval.status === "pending");
  const unknown = approvals.filter((approval) => approval.kind === "reconciliation" && approval.status === "unknown");
  return (
    <section className="review panel">
      <span className="section-label">RELEASE REVIEW</span>
      <h2>
        {unknown.length
          ? "A release failed after it started"
          : pending.length
            ? "A release is waiting for your decision"
            : "Review recorded"}
      </h2>
      {unknown.map((approval) => (
        <div className="approval" key={approval.action_id}>
          <p><b>{approval.tool}</b> · raised {approval.error ?? "an error"}; its outcome is unknown</p>
          <p className="objective">
            Check whether the project workspace contains this release, then record what happened. A release that did
            not take effect runs again after a new approval.
          </p>
          <label htmlFor={`reason-${approval.action_id}`}>Evidence (required)</label>
          <textarea id={`reason-${approval.action_id}`} value={reason} onChange={(event) => setReason(event.target.value)} />
          <div className="actions">
            <button type="button" disabled={busy || reason.trim().length < 3} onClick={() => reconcile(approval, "committed")}>
              It took effect
            </button>
            <button
              type="button"
              className="secondary"
              disabled={busy || reason.trim().length < 3}
              onClick={() => reconcile(approval, "not_committed")}
            >
              It did not take effect
            </button>
          </div>
        </div>
      ))}
      {pending.map((approval) => (
        <div className="approval" key={approval.action_id}>
          <p><b>{approval.tool}</b> · rule “{approval.policy_rule}”</p>
          <p className="objective">{String(approval.arguments.summary ?? JSON.stringify(approval.arguments))}</p>
          <label htmlFor={`reason-${approval.action_id}`}>Decision note (optional)</label>
          <textarea id={`reason-${approval.action_id}`} value={reason} onChange={(event) => setReason(event.target.value)} />
          <div className="actions">
            <button type="button" disabled={busy} onClick={() => decide(approval, "approve")}>Approve release</button>
            <button type="button" className="secondary" disabled={busy} onClick={() => decide(approval, "reject")}>Reject</button>
          </div>
        </div>
      ))}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function App() {
  const savedRunKey = `agloom:product-studio:${keycloak.tokenParsed?.tenant_id}:${keycloak.subject}:run`;
  const [projects, setProjects] = useState<Project[]>([]);
  const [selected, setSelected] = useState<Project | null>(null);
  const [objective, setObjective] = useState("Implement the seeded Pulseboard frontend and backend, then validate every criterion.");
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    authenticatedFetch("/api/projects").then((response) => response.json()).then((items: Project[]) => {
      setProjects(items);
      setSelected(items[0] ?? null);
    }).catch(() => setError("The studio API is unavailable."));
  }, []);

  useEffect(() => {
    const runId = sessionStorage.getItem(savedRunKey);
    if (!runId) return;
    const controller = new AbortController();
    let mounted = true;
    setRunning(true);
    authenticatedFetch(`/api/runs/${runId}`).then(async (response) => {
      if (response.status === 403 || response.status === 404) {
        sessionStorage.removeItem(savedRunKey);
        throw new Error("Saved run is no longer available for this account.");
      }
      if (!response.ok) throw new Error(await response.text());
      const current: Run = await response.json();
      if (mounted) {
        setRun(current);
        await streamRun(runId, controller);
      }
    }).catch((cause) => {
      if (mounted && !(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Run recovery failed");
      }
    }).finally(() => { if (mounted) setRunning(false); });
    return () => { mounted = false; controller.abort(); };
  }, [savedRunKey]);

  async function pollRun(runId: string) {
    let current: Run;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const response = await authenticatedFetch(`/api/runs/${runId}`);
      if (!response.ok) throw new Error(await response.text());
      current = await response.json();
      setRun(current);
    } while (ACTIVE.includes(current.status));
    return current;
  }

  function streamRun(runId: string, controller = new AbortController()) {
    return new Promise<Run>((resolve, reject) => {
      streamSse(`/api/runs/${runId}/events`, (event: RunEvent) => {
        setEvents((current) => [...current.slice(-39), event]);
        if (event.run) setRun(event.run);
        if (TERMINAL_EVENTS.includes(event.type)) {
          controller.abort();
          if (event.run) resolve(event.run);
          else pollRun(runId).then(resolve, reject);
        }
      }, controller.signal).then(() => {
        if (controller.signal.aborted) reject(new DOMException("Run stream closed", "AbortError"));
      }).catch((cause) => {
        if (controller.signal.aborted) reject(cause);
        else pollRun(runId).then(resolve, reject);
      });
    });
  }

  async function launch(event: FormEvent) {
    event.preventDefault();
    if (!selected) return;
    setRunning(true);
    setError("");
    setEvents([]);
    try {
      const taskResponse = await authenticatedFetch(`/api/projects/${selected.id}/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Build product", objective })
      });
      if (!taskResponse.ok) throw new Error(await taskResponse.text());
      const task: Task = await taskResponse.json();
      const runResponse = await authenticatedFetch(`/api/tasks/${task.id}/run`, {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
      });
      if (!runResponse.ok) throw new Error(await runResponse.text());
      const current: Run = await runResponse.json();
      setRun(current);
      sessionStorage.setItem(savedRunKey, current.id);
      await streamRun(current.id);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Run failed");
      }
    } finally {
      setRunning(false);
    }
  }

  async function cancel() {
    if (!run) return;
    const response = await authenticatedFetch(`/api/runs/${run.id}/cancel`, { method: "POST" });
    if (!response.ok) setError(await response.text());
  }

  return (
    <main>
      <header className="hero">
        <div><span className="kicker">AGLOOM · LOOP ENGINEERING</span><h1>Autonomous<br />Product Studio</h1></div>
        <p>Bounded software building with objective evidence and reviewed releases—not “looks done” model output.<br />
          Signed in as {keycloak.subject} · tenant {String(keycloak.tokenParsed?.tenant_id)}
          {" "}<button type="button" onClick={() => keycloak.logout()}>Logout</button>
        </p>
      </header>
      <section className="layout">
        <aside className="projects panel">
          <span className="section-label">PROJECTS</span>
          {projects.map((project) => (
            <button className={selected?.id === project.id ? "project active" : "project"} key={project.id} onClick={() => setSelected(project)}>
              <span>{project.name}</span><small>{project.acceptance_criteria.length} criteria</small>
            </button>
          ))}
        </aside>
        <section className="workspace panel">
          <span className="section-label">BOUNDED BUILD</span>
          <h2>{selected?.name ?? "Loading specification…"}</h2>
          <p className="objective">{selected?.objective}</p>
          <div className="criteria">
            {selected?.acceptance_criteria.map((criterion) => (
              <div key={criterion.id}><b>{criterion.id}</b><span>{criterion.description}</span><em>{criterion.kind}</em></div>
            ))}
          </div>
          <form onSubmit={launch}>
            <label htmlFor="objective">Iteration objective</label>
            <textarea id="objective" value={objective} onChange={(event) => setObjective(event.target.value)} />
            <button type="submit" disabled={running || !selected}>{running ? "Loop running…" : "Start bounded build →"}</button>
          </form>
          {error && <p className="error">{error}</p>}
        </section>
      </section>
      {run && (
        <section className="proof panel">
          <div>
            <span className="section-label">MEASURABLE PROOF</span>
            <h2>{HEADLINES[run.status]}</h2>
            <p>{run.proof.termination_reason ?? "Waiting for deterministic validation…"}</p>
            {ACTIVE.includes(run.status) && run.status !== "awaiting_reconciliation" && (
              <button type="button" className="secondary" onClick={cancel}>Cancel run</button>
            )}
          </div>
          <div className="metrics">
            <Metric label="Iterations" value={run.proof.iterations} />
            <Metric label="Criteria" value={`${run.proof.passed_criteria}/${run.proof.acceptance_criteria}`} />
            <Metric label="Pass rate" value={`${Math.round(run.proof.validation_pass_rate * 100)}%`} />
            <Metric label="Tool / model" value={`${run.proof.tool_calls} / ${run.proof.model_calls}`} />
            <Metric label="Elapsed" value={`${run.proof.elapsed_seconds.toFixed(1)}s`} />
          </div>
        </section>
      )}
      {run && WAITING.includes(run.status) && <ReviewPanel run={run} onDecided={setRun} />}
      {events.length > 0 && (
        <section className="stream panel" aria-live="polite">
          <div className="stream-heading">
            <div><span className="section-label">LIVE RUN STREAM</span><h2>Status &amp; progress</h2></div>
            <span className={running ? "stream-state active" : "stream-state"}>{running ? "CONNECTED" : "COMPLETE"}</span>
          </div>
          <ol>
            {events.map((event, index) => (
              <li key={`${event.id ?? event.timestamp}-${index}`}>
                <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                <b>{event.type === "runtime" ? event.name : event.type}</b>
                <span>{event.message ?? event.action ?? (event.iteration ? `Iteration ${event.iteration}` : "Runtime event")}</span>
                {event.progress !== undefined && <em>{Math.round(event.progress * 100)}%</em>}
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}

initializeAuth()
  .then(() => createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>))
  .catch((error) => { document.getElementById("root")!.textContent = `Authentication failed: ${String(error)}`; });
