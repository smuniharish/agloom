import React, { FormEvent, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Evidence = { claim: string; citation: string };
type Report = {
  summary: string;
  risk: string;
  evidence: Evidence[];
  recommendation: string;
  mcp_checks: Record<string, string>;
};

function App() {
  const [question, setQuestion] = useState(
    "Is checkout release 4.9 ready to deploy, and what evidence supports the decision?"
  );
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      if (!response.ok) throw new Error(await response.text());
      setReport(await response.json());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Investigation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <span className="eyebrow">Agloom · live evidence workspace</span>
        <h1>Engineering Change Investigator</h1>
        <p>Correlate change records, runbooks, live web status, and MCP evidence.</p>
      </header>
      <section className="panel">
        <form onSubmit={submit}>
          <label htmlFor="question">Release question</label>
          <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button disabled={loading}>{loading ? "Investigating…" : "Run investigation"}</button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>
      {report && (
        <section className="results">
          <article className="summary">
            <span className="risk">{report.risk}</span>
            <h2>Decision</h2>
            <p>{report.summary}</p>
            <h3>Recommendation</h3>
            <p>{report.recommendation}</p>
          </article>
          <article>
            <h2>Grounded evidence</h2>
            {report.evidence.map((item, index) => (
              <div className="evidence" key={index}>
                <strong>{item.citation}</strong><p>{item.claim}</p>
              </div>
            ))}
          </article>
          <article>
            <h2>MCP verification</h2>
            <div className="checks">
              {Object.entries(report.mcp_checks).map(([name, status]) => (
                <span key={name}>✓ {name}: {status}</span>
              ))}
            </div>
          </article>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
