import { useState, type FormEvent } from 'react'

type Audit = { audit_id: string; report: { verdict: string; summary: string; findings: { policy_id: string; requirement: string; status: string; rationale: string }[] }; evidence: { policy_id: string; title: string; excerpt: string }[] }
export default function App() {
  const [url, setUrl] = useState('http://target-site')
  const [audit, setAudit] = useState<Audit | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('')
    try { const response = await fetch('/api/audits', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({url}) }); if (!response.ok) throw new Error(await response.text()); setAudit(await response.json()) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Audit failed') } finally { setBusy(false) }
  }
  return <main><header><span className="mark">✓</span><div><h1>Web Compliance Auditor</h1><p>Grounded URL audit against your approved local policies</p></div></header><section className="card"><h2>Audit a URL</h2><form onSubmit={submit}><label htmlFor="url">URL to inspect with Microsoft Playwright MCP</label><div className="row"><input id="url" type="url" value={url} onChange={e=>setUrl(e.target.value)} required /><button disabled={busy}>{busy ? 'Auditing…' : 'Run audit'}</button></div></form>{error && <p role="alert" className="error">{error}</p>}<p className="note"><code>http://target-site</code> is the intentionally noncompliant site bundled with the Compose stack. Evidence is retrieved from PostgreSQL + pgvector with local Ollama all-minilm vectors. Results remain reviewable, not legal advice.</p></section>{audit && <><section className={`card verdict ${audit.report.verdict}`}><h2>{audit.report.verdict.replace('_',' ')}</h2><p>{audit.report.summary}</p></section><section className="grid"><article className="card"><h2>Findings</h2>{audit.report.findings.map((f,i)=><div className="finding" key={i}><b>{f.status}</b><h3>{f.requirement}</h3><small>{f.policy_id}</small><p>{f.rationale}</p></div>)}</article><article className="card"><h2>Retrieved policy evidence</h2>{audit.evidence.map(e=><details key={e.policy_id}><summary>{e.policy_id} — {e.title}</summary><p>{e.excerpt}</p></details>)}</article></section></>}</main>
}
