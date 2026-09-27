import { useEffect, useMemo, useRef, useState } from 'react'
import {
  analyzeIncident,
  checkHealth,
  isRecord,
  listIncidents,
  type Citation,
  type Evidence,
  type Incident,
  type RcaReport,
} from './api'

function Icon({ name, size = 18 }: { name: 'spark' | 'arrow' | 'search' | 'refresh' | 'check' | 'link' | 'clock' | 'alert' | 'layers' | 'shield'; size?: number }) {
  const paths: Record<typeof name, React.ReactNode> = {
    spark: <><path d="m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Z" /><path d="m19 17 .6 2.4L22 20l-2.4.6L19 23l-.6-2.4L16 20l2.4-.6L19 17ZM4 16l.7 2.3L7 19l-2.3.7L4 22l-.7-2.3L1 19l2.3-.7L4 16Z" /></>,
    arrow: <><path d="M4 12h15m-6-6 6 6-6 6" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m16 16 5 5" /></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5.6 9a7 7 0 0 1 12-2L20 12M4 12l2.4 5a7 7 0 0 0 12-2" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    link: <><path d="M10 13a5 5 0 0 0 7 .3l3-3a5 5 0 0 0-7-7l-1.7 1.7M14 11a5 5 0 0 0-7-.3l-3 3a5 5 0 0 0 7 7l1.7-1.7" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    alert: <><path d="m12 3 10 18H2L12 3Z" /><path d="M12 9v5m0 3h.01" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5" /></>,
    shield: <><path d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3Z" /><path d="m9 12 2 2 4-4" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function formatDate(value?: string) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' }).format(date)
}

function confidence(value?: number | string) {
  if (typeof value === 'number' && Number.isFinite(value)) return `${Math.round(value <= 1 ? value * 100 : value)}%`
  return value == null ? null : String(value)
}

function sentence(value: unknown): string {
  if (typeof value === 'string') return value
  if (isRecord(value)) return String(value.description ?? value.summary ?? value.claim ?? value.title ?? value.action ?? '')
  return ''
}

function safeUrl(value?: string) {
  if (!value) return null
  try {
    const url = new URL(value)
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : null
  } catch {
    return null
  }
}

function CitationTags({ items, catalog = [] }: { items?: Array<Citation | string>; catalog?: Citation[] }) {
  if (!items?.length) return null
  return <div className="citation-tags" aria-label="Sources">{items.map((item, index) => {
    const citation = typeof item === 'string' ? catalog.find((entry) => entry.id === item) ?? { id: item } : item
    const label = citation.title || citation.source || citation.id || `Source ${index + 1}`
    const url = safeUrl(citation.url)
    return url
      ? <a className="citation-tag" key={`${label}-${index}`} href={url} target="_blank" rel="noopener noreferrer"><Icon name="link" size={12} />{label}<span className="external-indicator" aria-hidden="true">↗</span></a>
      : <span className="citation-tag" key={`${label}-${index}`}><Icon name="link" size={12} />{label}</span>
  })}</div>
}

function Report({ report, incident }: { report: RcaReport; incident: Incident }) {
  const root = report.root_cause
  const rootConfidence = isRecord(root) ? confidence(root.confidence as number | string | undefined) : null
  const rootCitations = isRecord(root) && Array.isArray(root.citations) ? root.citations as Array<Citation | string> : undefined
  const evidence = Array.isArray(report.evidence) ? report.evidence : []
  const citations = Array.isArray(report.citations) ? report.citations : []
  const factors = Array.isArray(report.contributing_factors) ? report.contributing_factors : []
  const timeline = Array.isArray(report.timeline) ? report.timeline : []
  const recommendations = Array.isArray(report.recommendations) ? report.recommendations : []
  const score = confidence(report.confidence)

  return <div className="report" aria-label="Root cause analysis report">
    <div className="report-header">
      <div><div className="eyebrow"><span className="eyebrow-line" /> ANALYSIS COMPLETE</div><h2>Root cause analysis</h2><p>Evidence-led findings for {incident.title}</p></div>
      <span className="complete-badge"><Icon name="check" size={14} /> Ready to review</span>
    </div>

    <div className="report-overview">
      <div className="report-overview-main"><div className="section-kicker">THE SHORT VERSION</div><h3>{report.summary || sentence(root) || 'Analysis complete'}</h3><p>{report.summary && sentence(root) && report.summary !== sentence(root) ? sentence(root) : 'Review the findings and supporting sources below.'}</p></div>
      <div className="confidence-panel"><span className="confidence-symbol"><Icon name="shield" size={23} /></span><span className="confidence-label">Overall confidence</span><strong>{score ?? 'Not rated'}</strong><span className="confidence-caption">{score ? 'Based on available evidence' : 'No confidence score provided'}</span></div>
    </div>

    <div className="report-grid">
      <div className="report-primary">
        <section className="report-section root-section">
          <div className="section-heading"><span className="section-icon purple"><Icon name="spark" size={18} /></span><div><span className="section-kicker">01 / FINDINGS</span><h3>Root cause</h3></div></div>
          <p className="section-text">{sentence(root) || 'A root cause was not included in this analysis.'}</p>
          {rootConfidence && <span className="mini-confidence">Confidence · {rootConfidence}</span>}
          <CitationTags items={rootCitations} catalog={citations} />
        </section>

        {report.impact && <section className="report-section"><div className="section-heading"><span className="section-icon amber"><Icon name="alert" size={18} /></span><div><span className="section-kicker">02 / SCOPE</span><h3>Impact</h3></div></div><p className="section-text">{report.impact}</p></section>}

        {factors.length > 0 && <section className="report-section"><div className="section-heading"><span className="section-icon blue"><Icon name="layers" size={18} /></span><div><span className="section-kicker">CONTEXT</span><h3>Contributing factors</h3></div></div><ul className="factor-list">{factors.map((factor, index) => <li key={index}><span className="factor-number">{String(index + 1).padStart(2, '0')}</span><div><span>{sentence(factor)}</span>{isRecord(factor) && <CitationTags items={Array.isArray(factor.citations) ? factor.citations as Array<Citation | string> : factor.citation_ids as string[] | undefined} catalog={citations} />}</div></li>)}</ul></section>}

        {evidence.length > 0 && <section className="report-section"><div className="section-heading"><span className="section-icon green"><Icon name="shield" size={18} /></span><div><span className="section-kicker">TRACEABLE FINDINGS</span><h3>Supporting evidence <span className="heading-count">{evidence.length}</span></h3></div></div><div className="evidence-list">{evidence.map((item: Evidence, index) => <article className="evidence-item" key={index}><div className="evidence-top"><span className="evidence-index">E{String(index + 1).padStart(2, '0')}</span>{confidence(item.confidence) && <span className="mini-confidence">{confidence(item.confidence)} confidence</span>}</div><p>{sentence(item)}</p>{item.source && <small>Source: {item.source}</small>}<CitationTags items={item.citations ?? item.citation_ids} catalog={citations} /></article>)}</div></section>}

        {recommendations.length > 0 && <section className="report-section"><div className="section-heading"><span className="section-icon purple"><Icon name="arrow" size={18} /></span><div><span className="section-kicker">NEXT STEPS</span><h3>Recommended actions</h3></div></div><div className="action-list">{recommendations.map((item, index) => <div className="action-item" key={index}><span className="action-check"><Icon name="check" size={14} /></span><div><strong>{typeof item === 'string' ? item : item.title || item.action || item.description}</strong>{typeof item !== 'string' && item.title && item.description && <p>{item.description}</p>}</div>{typeof item !== 'string' && item.priority && <span className="priority-tag">{item.priority}</span>}</div>)}</div></section>}
      </div>

      <aside className="report-aside">
        {timeline.length > 0 && <section className="aside-section"><div className="aside-title"><Icon name="clock" size={17} /><h3>Incident timeline</h3></div><div className="timeline">{timeline.map((item, index) => <div className="timeline-item" key={index}><span className="timeline-dot" /><time>{formatDate(item.timestamp ?? item.time) || `Event ${index + 1}`}</time><strong>{item.title || item.event || item.description || 'Incident event'}</strong>{(item.title || item.event) && item.description && <p>{item.description}</p>}<CitationTags items={item.citations} catalog={citations} /></div>)}</div></section>}
        {citations.length > 0 && <section className="aside-section"><div className="aside-title"><Icon name="link" size={17} /><h3>Sources <span className="heading-count">{citations.length}</span></h3></div><div className="source-list">{citations.map((source, index) => {
          const url = safeUrl(source.url)
          return <div className="source-item" key={source.id ?? index}><span className="source-number">{String(index + 1).padStart(2, '0')}</span><div>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{source.title || source.source || source.id || `Source ${index + 1}`} <span aria-hidden="true">↗</span></a> : <strong>{source.title || source.source || source.id || `Source ${index + 1}`}</strong>}{source.excerpt && <p>{source.excerpt}</p>}{source.timestamp && <small>{formatDate(source.timestamp)}</small>}</div></div>
        })}</div></section>}
        {report.generated_at && <div className="generated-note"><Icon name="clock" size={15} /> Generated {formatDate(report.generated_at)}</div>}
      </aside>
    </div>
  </div>
}

export default function App() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [query, setQuery] = useState('')
  const [loadingList, setLoadingList] = useState(true)
  const [listError, setListError] = useState('')
  const [analysisError, setAnalysisError] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [report, setReport] = useState<RcaReport | null>(null)
  const [health, setHealth] = useState<'checking' | 'online' | 'unconfigured' | 'offline'>('checking')
  const analysisController = useRef<AbortController | null>(null)
  const listController = useRef<AbortController | null>(null)

  function loadIncidents() {
    listController.current?.abort()
    const controller = new AbortController()
    listController.current = controller
    setLoadingList(true)
    setListError('')
    void listIncidents(controller.signal).then((rows) => {
      setIncidents(rows)
      setSelectedId((current) => rows.some((row) => row.id === current) ? current : '')
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setListError(error instanceof Error ? error.message : 'Could not load incidents.')
    }).finally(() => {
      if (!controller.signal.aborted) setLoadingList(false)
    })
  }

  useEffect(() => {
    loadIncidents()
    const controller = new AbortController()
    void checkHealth(controller.signal).then(setHealth).catch(() => {
      if (!controller.signal.aborted) setHealth('offline')
    })
    return () => {
      controller.abort()
      listController.current?.abort()
      analysisController.current?.abort()
    }
  }, [])

  const selected = incidents.find((incident) => incident.id === selectedId)
  const filtered = useMemo(() => incidents.filter((incident) => `${incident.title} ${incident.id} ${incident.service ?? ''}`.toLowerCase().includes(query.toLowerCase().trim())), [incidents, query])

  function selectIncident(id: string) {
    if (id === selectedId) return
    analysisController.current?.abort()
    setSelectedId(id)
    setReport(null)
    setAnalysisError('')
    setAnalyzing(false)
  }

  async function runAnalysis() {
    if (!selectedId || analyzing) return
    const controller = new AbortController()
    analysisController.current = controller
    setAnalyzing(true)
    setReport(null)
    setAnalysisError('')
    try {
      const result = await analyzeIncident(selectedId, controller.signal)
      if (!controller.signal.aborted) setReport(result)
    } catch (error) {
      if (!controller.signal.aborted) setAnalysisError(error instanceof Error ? error.message : 'Analysis failed. Please try again.')
    } finally {
      if (!controller.signal.aborted) setAnalyzing(false)
    }
  }

  return <div className="app-shell">
    <header className="topbar"><div className="topbar-inner"><a className="brand" href="/" aria-label="RCA Generator home"><span className="brand-mark"><Icon name="spark" size={19} /></span><span>root<span className="brand-accent">cause</span><span className="brand-dot">.</span></span></a><span className="topbar-divider" /><span className="topbar-label">Incident intelligence</span><div className="topbar-right"><span className={`health-indicator ${health}`}><span className="health-dot" />{health === 'online' ? 'System connected' : health === 'unconfigured' ? 'Model not configured' : health === 'offline' ? 'API unavailable' : 'Checking connection'}</span><span className="version-tag">WORKSPACE</span></div></div></header>

    <main className="workspace">
      <div className="page-intro"><div className="breadcrumb">WORKSPACE <span>/</span> INCIDENT ANALYSIS</div><div className="intro-row"><div><h1>Understand what <em>happened.</em></h1><p>Turn incident data into a clear, evidence-backed root cause analysis.</p></div><div className="intro-decoration" aria-hidden="true"><span /><span /><span /></div></div></div>

      <div className="workspace-grid"><aside className="sidebar" aria-label="Incident selection"><div className="sidebar-heading"><div><span className="section-kicker">YOUR WORKSPACE</span><h2>Incidents <span className="heading-count">{incidents.length}</span></h2></div><button className="icon-button" type="button" title="Refresh incidents" aria-label="Refresh incidents" onClick={loadIncidents} disabled={loadingList}><Icon name="refresh" size={17} /></button></div>
        <label className="search-field"><Icon name="search" size={17} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search incidents..." aria-label="Search incidents" /></label>
        <div className="incident-list">
          {loadingList && <div className="sidebar-message"><span className="spinner small" /> Loading incidents…</div>}
          {!loadingList && listError && <div className="sidebar-error" role="alert"><Icon name="alert" size={18} /><span>{listError}</span><button type="button" onClick={loadIncidents}>Retry</button></div>}
          {!loadingList && !listError && filtered.length === 0 && <div className="sidebar-message">{query ? 'No matching incidents.' : 'No incidents available yet.'}</div>}
          {!loadingList && !listError && filtered.map((incident) => <button className={`incident-card ${selectedId === incident.id ? 'selected' : ''}`} key={incident.id} type="button" onClick={() => selectIncident(incident.id)} aria-pressed={selectedId === incident.id}><span className="incident-card-top"><span className="incident-id">{incident.id}</span>{incident.severity && <span className={`severity ${/critical|sev.?1|high/i.test(incident.severity) ? 'high' : ''}`}>{incident.severity}</span>}</span><strong>{incident.title}</strong>{incident.description && <span className="incident-description">{incident.description}</span>}<span className="incident-meta">{incident.service && <span>{incident.service}</span>}{incident.started_at && <span>{formatDate(incident.started_at)}</span>}</span></button>)}
        </div><div className="sidebar-footer"><Icon name="shield" size={16} /> Analysis grounded in your incident data</div>
      </aside>

      <div className="content-area"><section className="selection-panel"><div className="selection-info"><span className="step-label"><span>01</span> SELECT AN INCIDENT</span><h2>{selected ? selected.title : 'Choose an incident to investigate'}</h2><p>{selected ? selected.description || `Incident ${selected.id}${selected.service ? ` · ${selected.service}` : ''}` : 'Pick an incident from the list to begin your investigation.'}</p>{selected && <div className="selection-meta"><span>{selected.id}</span>{selected.status && <span>{selected.status}</span>}{selected.started_at && <span>{formatDate(selected.started_at)}</span>}</div>}</div><button type="button" className="analyze-button" disabled={!selected || analyzing} onClick={() => void runAnalysis()}>{analyzing ? <><span className="spinner" /> Analyzing…</> : <><Icon name="spark" size={17} /> Generate analysis <Icon name="arrow" size={17} /></>}</button></section>
        {analysisError && <div className="error-banner" role="alert"><Icon name="alert" size={20} /><div><strong>Analysis couldn't be completed</strong><p>{analysisError}</p></div><button type="button" onClick={() => void runAnalysis()}>Try again</button></div>}
        {analyzing && <div className="loading-panel" role="status"><div className="loading-art"><div className="loading-orbit"><Icon name="spark" size={29} /></div></div><div className="eyebrow">CONNECTING THE DOTS</div><h2>Building your analysis</h2><p>Reviewing incident signals, tracing evidence, and assembling the findings. This may take a moment.</p><div className="loading-track"><span /></div></div>}
        {!analyzing && report && selected && <Report report={report} incident={selected} />}
        {!analyzing && !report && !analysisError && <div className="empty-panel"><div className="empty-illustration"><span className="empty-halo halo-one" /><span className="empty-halo halo-two" /><span className="empty-core"><Icon name="layers" size={31} /></span><span className="empty-node node-one" /><span className="empty-node node-two" /><span className="empty-node node-three" /></div><span className="section-kicker">CLARITY STARTS HERE</span><h2>From incident to insight.</h2><p>{selected ? 'Your incident is ready. Generate an analysis to uncover the root cause, trace the evidence, and plan the way forward.' : 'Select an incident on the left, then generate an analysis to uncover what went wrong and why.'}</p><div className="empty-features"><span><Icon name="check" size={14} /> Root cause</span><span><Icon name="check" size={14} /> Evidence & sources</span><span><Icon name="check" size={14} /> Next steps</span></div></div>}
      </div></div>
      <footer className="page-footer">ROOTCAUSE <span>·</span> Evidence over assumptions</footer>
    </main>
  </div>
}
