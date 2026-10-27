import { useState, type FormEvent } from "react";

type Verdict = "pass" | "fail" | "needs_review";
type Finding = {
  policy_id: string;
  requirement: string;
  status: Verdict;
  rationale: string;
  evidence_ids: string[];
};
type Evidence = { policy_id: string; title: string; excerpt: string; similarity: number };
type ComplianceRecord = {
  status: "recorded" | "awaiting_approval" | "rejected" | "expired" | "outcome_unknown";
  action_id: string | null;
  policy_rule: string | null;
  decided_by: string | null;
  recorded_at: string | null;
};
type Audit = {
  audit_id: string;
  url: string;
  report: { verdict: Verdict; summary: string; findings: Finding[]; limitations: string[] };
  evidence: Evidence[];
  mcp_checks: Record<string, string>;
  record: ComplianceRecord;
  created_at: string;
};

const RECORD_TEXT: Record<ComplianceRecord["status"], string> = {
  recorded: "Recorded in the compliance register",
  awaiting_approval: "Waiting for a compliance officer",
  rejected: "Rejected by a compliance officer",
  expired: "Approval window expired",
  outcome_unknown: "Record outcome unknown: an operator must reconcile it"
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

export default function App() {
  const [url, setUrl] = useState("http://target-site");
  const [audit, setAudit] = useState<Audit | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [correction, setCorrection] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The request failed");
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void run(async () => {
      setNotice("");
      setAudit(await postJson<Audit>("/api/audits", { url }));
    });
  }

  function review(approve: boolean) {
    if (!audit) return;
    void run(async () => {
      setAudit(
        await postJson<Audit>(`/api/audits/${audit.audit_id}/approval`, { approve, reviewer })
      );
    });
  }

  function recordReconciled() {
    if (!audit) return;
    void run(async () => {
      setAudit(await postJson<Audit>(`/api/audits/${audit.audit_id}/record`, {}));
    });
  }

  function rate(rating: "up" | "down") {
    if (!audit) return;
    void run(async () => {
      await postJson(`/api/audits/${audit.audit_id}/feedback`, {
        rating,
        correction: correction.trim() || null
      });
      setNotice("Thanks, your feedback was recorded.");
    });
  }

  return (
    <main>
      <header>
        <span className="mark">✓</span>
        <div>
          <h1>Web Compliance Auditor</h1>
          <p>Grounded URL audits against your approved policies</p>
        </div>
      </header>
      <section className="card">
        <h2>Audit a URL</h2>
        <form onSubmit={submit}>
          <label htmlFor="url">URL to inspect with Playwright MCP</label>
          <div className="row">
            <input id="url" type="url" value={url} onChange={(e) => setUrl(e.target.value)} required />
            <button disabled={busy}>{busy ? "Working…" : "Run audit"}</button>
          </div>
        </form>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <p className="note">
          <code>http://target-site</code> is the intentionally noncompliant page of the Compose
          stack. Policies are retrieved from PostgreSQL with pgvector. Results are reviewable, not
          legal advice.
        </p>
      </section>
      {audit && (
        <>
          <section className={`card verdict ${audit.report.verdict}`}>
            <h2>{audit.report.verdict.replace("_", " ")}</h2>
            <p>{audit.report.summary}</p>
            {audit.report.limitations.map((limitation) => (
              <p className="note" key={limitation}>
                {limitation}
              </p>
            ))}
          </section>
          <section className="grid">
            <article className="card">
              <h2>Findings</h2>
              {audit.report.findings.map((finding, index) => (
                <div className={`finding ${finding.status}`} key={index}>
                  <b>{finding.status.replace("_", " ")}</b>
                  <h3>{finding.requirement}</h3>
                  <small>{finding.policy_id}</small>
                  <p>{finding.rationale}</p>
                </div>
              ))}
            </article>
            <article className="card">
              <h2>Compliance register</h2>
              <p className={`status ${audit.record.status}`}>{RECORD_TEXT[audit.record.status]}</p>
              {audit.record.policy_rule && <p className="note">Policy rule: {audit.record.policy_rule}</p>}
              {audit.record.decided_by && <p className="note">Reviewed by {audit.record.decided_by}</p>}
              {audit.record.status === "outcome_unknown" && audit.record.action_id && (
                <div className="review">
                  <p className="note">agentbarrier action: {audit.record.action_id}</p>
                  <div className="row">
                    <button disabled={busy} onClick={recordReconciled}>
                      Record after reconciliation
                    </button>
                  </div>
                </div>
              )}
              {audit.record.status === "awaiting_approval" && (
                <div className="review">
                  <label htmlFor="reviewer">Compliance officer</label>
                  <input
                    id="reviewer"
                    value={reviewer}
                    onChange={(e) => setReviewer(e.target.value)}
                    placeholder="Your name"
                  />
                  <div className="row">
                    <button disabled={busy || !reviewer.trim()} onClick={() => review(true)}>
                      Certify
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
              <h2>MCP verification</h2>
              <ul className="checks">
                {Object.entries(audit.mcp_checks).map(([name, status]) => (
                  <li key={name}>
                    {name}: {status}
                  </li>
                ))}
              </ul>
            </article>
          </section>
          <section className="grid">
            <article className="card">
              <h2>Retrieved policy evidence</h2>
              {audit.evidence.map((item) => (
                <details key={item.policy_id}>
                  <summary>
                    {item.policy_id} — {item.title}
                  </summary>
                  <p>{item.excerpt}</p>
                </details>
              ))}
            </article>
            <article className="card">
              <h2>Feedback</h2>
              <label htmlFor="correction">Correction (optional)</label>
              <textarea
                id="correction"
                value={correction}
                onChange={(e) => setCorrection(e.target.value)}
              />
              <div className="row">
                <button className="secondary" disabled={busy} onClick={() => rate("up")}>
                  Helpful
                </button>
                <button className="secondary" disabled={busy} onClick={() => rate("down")}>
                  Not helpful
                </button>
              </div>
              {notice && <p className="note">{notice}</p>}
            </article>
          </section>
        </>
      )}
    </main>
  );
}
