export interface Citation {
  id: string
  document_id: string
  title: string
  heading: string
  owner: string
  updated: string
  excerpt: string
}

export interface Answer {
  conversation_id: string
  answer: string
  citations: Citation[]
}

export interface Document {
  id: string
  title: string
  owner: string
  updated: string
}

const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${base}${path}`, init)
  } catch {
    throw new Error('Could not reach the knowledge service. Is the backend running?')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null && 'detail' in body
      ? body.detail : null
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return body as T
}

export function listDocuments(): Promise<Document[]> {
  return request<Document[]>('/documents')
}

export function ask(question: string, conversationId: string | null): Promise<Answer> {
  return request<Answer>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  })
}
