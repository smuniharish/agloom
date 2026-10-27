import React, { FormEvent, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Evidence = { claim: string; citation: string };
type Report = {
  decision: "go" | "no_go";
  risk: "low" | "medium" | "high" | "critical";
  summary: string;
  evidence: Evidence[];
  recommendation: string;
};
type ReleaseDecision = {
  status: "recorded" | "awaiting_approval" | "rejected" | "expired" | "outcome_unknown";
  action_id: string | null;
  policy_rule: string | null;
  decided_by: string | null;
  recorded_at: string | null;
};
type Investigation = {
  investigation_id: string;
  question: string;
  report: Report;
  mcp_checks: Record<string, string>;
  release_decision: ReleaseDecision;
  created_at: string;
};

const STATUS_TEXT: Record<ReleaseDecision["status"], string> = {
  recorded: "Published to the release register",
  awaiting_approval: "Waiting for a release manager",
  rejected: "Rejected by a release manager",
  expired: "Approval window expired",
  outcome_unknown: "Publication outcome unknown: an operator must reconcile it"
};

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

function App() {
  const [question, setQuestion] = useState(
    "Is checkout release 4.9 ready to deploy, and what evidence supports the decision?"
  );
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The request failed");
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void run(async () => {
      setFeedback("");
      setInvestigation(await postJson<Investigation>("/api/investigate", { question }));
    });
  }

  function review(approve: boolean) {
    if (!investigation) return;
    void run(async () => {
      setInvestigation(
        await postJson<Investigation>(
          `/api/investigations/${investigation.investigation_id}/approval`,
          { approve, reviewer }
        )
      );
    });
  }

  function publishReconciled() {
    if (!investigation) return;
    void run(async () => {
      setInvestigation(
        await postJson<Investigation>(
          `/api/investigations/${investigation.investigation_id}/publication`,
          {}
        )
      );
    });
  }

  function rate(rating: "up" | "down") {
    if (!investigation) return;
    void run(async () => {
      await postJson(`/api/investigations/${investigation.investigation_id}/feedback`, { rating });
      setFeedback("Thanks, your feedback was recorded.");
    });
  }

  const report = investigation?.report;
  const decision = investigation?.release_decision;

  return (
    <main>
      <header>
        <span className="eyebrow">Agloom · governed release decisions</span>
        <h1>Engineering Change Investigator</h1>
        <p>Correlate change records, runbooks, live release status, and MCP evidence.</p>
      </header>
      <section className="panel">
        <form onSubmit={submit}>
          <label htmlFor="question">Release question</label>
          <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button disabled={busy}>{busy ? "Working…" : "Run investigation"}</button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
      </section>
      {report && decision && investigation && (
        <section className="results">
          <article className="summary">
            <span className={`verdict ${report.decision}`}>
              {report.decision === "go" ? "Go" : "No go"}
            </span>
            <span className="risk">{report.risk} risk</span>
            <h2>Decision</h2>
            <p>{report.summary}</p>
            <h3>Recommendation</h3>
            <p>{report.recommendation}</p>
          </article>
          <article>
            <h2>Grounded evidence</h2>
            {report.evidence.map((item, index) => (
              <div className="evidence" key={index}>
                <strong>{item.citation}</strong>
                <p>{item.claim}</p>
              </div>
            ))}
          </article>
          <article>
            <h2>Release register</h2>
            <p className={`status ${decision.status}`}>{STATUS_TEXT[decision.status]}</p>
            {decision.policy_rule && <p className="note">Policy rule: {decision.policy_rule}</p>}
            {decision.decided_by && <p className="note">Reviewed by {decision.decided_by}</p>}
            {decision.status === "outcome_unknown" && decision.action_id && (
              <div className="review">
                <p className="note">agentbarrier action: {decision.action_id}</p>
                <div className="actions">
                  <button disabled={busy} onClick={publishReconciled}>
                    Publish after reconciliation
                  </button>
                </div>
              </div>
            )}
            {decision.status === "awaiting_approval" && (
              <div className="review">
                <label htmlFor="reviewer">Release manager</label>
                <input
                  id="reviewer"
                  value={reviewer}
                  onChange={(e) => setReviewer(e.target.value)}
                  placeholder="Your name"
                />
                <div className="actions">
                  <button disabled={busy || !reviewer.trim()} onClick={() => review(true)}>
                    Approve and publish
                  </button>
                  <button
                    className="secondary"
                    disabled={busy || !reviewer.trim()}
                    onClick={() => review(false)}
                  >
                    Reject
                  </button>
                </div>
              </div>
            )}
          </article>
          <article className="wide">
            <h2>MCP verification</h2>
            <div className="checks">
              {Object.entries(investigation.mcp_checks).map(([name, status]) => (
                <span key={name}>✓ {name}: {status}</span>
              ))}
            </div>
            <div className="actions">
              <button className="secondary" disabled={busy} onClick={() => rate("up")}>
                Helpful
              </button>
              <button className="secondary" disabled={busy} onClick={() => rate("down")}>
                Not helpful
              </button>
            </div>
            {feedback && <p className="note">{feedback}</p>}
          </article>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
