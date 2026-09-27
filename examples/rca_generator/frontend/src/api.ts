export interface Incident {
  id: string
  title: string
  description?: string
  severity?: string
  service?: string
  status?: string
  started_at?: string
}

export interface Citation {
  id?: string
  source?: string
  title?: string
  url?: string
  excerpt?: string
  timestamp?: string
  confidence?: number | string
}

export interface Evidence {
  claim?: string
  summary?: string
  description?: string
  source?: string
  confidence?: number | string
  citation_ids?: string[]
  citations?: Array<Citation | string>
}

export interface TimelineEvent {
  timestamp?: string
  time?: string
  title?: string
  event?: string
  description?: string
  citations?: Array<Citation | string>
}

export interface Recommendation {
  title?: string
  action?: string
  description?: string
  priority?: string
}

export interface RcaReport {
  incident_id?: string
  summary?: string
  root_cause?: string | { description?: string; summary?: string; confidence?: number | string; citations?: Array<Citation | string> }
  confidence?: number | string
  impact?: string
  contributing_factors?: Array<string | Evidence>
  evidence?: Evidence[]
  citations?: Citation[]
  timeline?: TimelineEvent[]
  recommendations?: Array<string | Recommendation>
  generated_at?: string
}

const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim() || '/api'
const apiBase = configuredBase.replace(/\/+$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new Error('Unable to reach the API. Check that the backend is running and try again.')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = isRecord(body) ? body.detail ?? body.error ?? body.message : null
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}). Please try again.`)
  }
  return body as T
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export async function listIncidents(signal?: AbortSignal): Promise<Incident[]> {
  const result = await request<unknown>('/incidents', { signal })
  const rows = Array.isArray(result) ? result : isRecord(result) ? result.incidents : null
  if (!Array.isArray(rows)) throw new Error('The incident list returned an unexpected format.')
  return rows.filter(isRecord).map((row) => ({
    id: String(row.id ?? row.incident_id ?? ''),
    title: String(row.title ?? row.name ?? row.summary ?? row.id ?? row.incident_id ?? 'Untitled incident'),
    description: text(row.description),
    severity: text(row.severity),
    service: text(row.service),
    status: text(row.status),
    started_at: text(row.started_at ?? row.created_at),
  })).filter((row) => row.id)
}

export async function analyzeIncident(incidentId: string, signal?: AbortSignal): Promise<RcaReport> {
  const result = await request<unknown>('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ incident_id: incidentId }),
    signal,
  })
  if (!isRecord(result)) throw new Error('The analysis returned an unexpected format.')
  const report = isRecord(result.report) ? result.report : isRecord(result.analysis) ? result.analysis : result
  return report as RcaReport
}

export async function checkHealth(signal?: AbortSignal): Promise<'online' | 'unconfigured'> {
  const response = await request<unknown>('/health', { signal })
  if (!isRecord(response)) throw new Error('Unexpected health response.')
  if (response.status === 'ok') return 'online'
  if (response.status === 'unconfigured') return 'unconfigured'
  throw new Error('API is not ready.')
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}
