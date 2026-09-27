import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ask, listDocuments, type Citation, type Document } from './api'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

const suggestions = [
  'How do I report a lost company device?',
  'What approvals are required before a production deployment?',
  'When should I submit an expense claim?',
]

function AnswerText({ children }: { children: string }) {
  return children.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={index}>{part.slice(2, -2)}</strong>
      : part,
  )
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [documentError, setDocumentError] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listDocuments().then(setDocuments).catch((reason: unknown) => {
      setDocumentError(reason instanceof Error ? reason.message : 'Unable to load documents')
    })
  }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function submit(value: string) {
    const text = value.trim()
    if (!text || busy) return
    setError('')
    setQuestion('')
    setBusy(true)
    setMessages(previous => [...previous, { role: 'user', content: text }])
    try {
      const result = await ask(text, conversationId)
      setConversationId(result.conversation_id)
      setMessages(previous => [...previous, {
        role: 'assistant', content: result.answer, citations: result.citations,
      }])
    } catch (reason) {
      setMessages(previous => previous.slice(0, -1))
      setQuestion(text)
      setError(reason instanceof Error ? reason.message : 'Unable to answer')
    } finally {
      setBusy(false)
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submit(question)
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">K</span><div><strong>Knowledge Desk</strong><small>Enterprise assistant</small></div></div>
        <button className="new-chat" onClick={() => { setConversationId(null); setMessages([]); setError('') }}>+ New conversation</button>
        <div className="sidebar-label">KNOWLEDGE BASE</div>
        {documentError && <p className="error" role="alert">{documentError}</p>}
        <ul className="documents">{documents.map(doc => <li key={doc.id}><span className="doc-icon">▤</span><div><strong>{doc.title}</strong><small>{doc.owner} · {doc.updated}</small></div></li>)}</ul>
        <p className="sidebar-foot">Answers are grounded in the documents shown here. Verify important decisions with the document owner.</p>
      </aside>
      <main className="main">
        <header><div><span className="online-dot" /> Internal knowledge</div><span>{documents.length} documents indexed</span></header>
        <section className="conversation" aria-label="Conversation">
          {messages.length === 0 && <div className="welcome">
            <div className="welcome-icon">✳</div>
            <p className="eyebrow">YOUR WORKPLACE, EXPLAINED</p>
            <h1>What can we help you find?</h1>
            <p>Ask a question about company procedures. Each answer includes the exact document passages it used.</p>
            <div className="suggestions">{suggestions.map(item => <button key={item} onClick={() => void submit(item)} disabled={busy}>{item}<span>↗</span></button>)}</div>
          </div>}
          <div className="messages">{messages.map((message, index) => <article className={`message ${message.role}`} key={index}>
            <div className="avatar">{message.role === 'assistant' ? 'K' : 'You'}</div>
            <div className="message-body"><strong>{message.role === 'assistant' ? 'Knowledge Desk' : 'You'}</strong><p><AnswerText>{message.content}</AnswerText></p>
              {message.citations && message.citations.length > 0 && <div className="sources"><span>Sources</span>{message.citations.map(source => <details key={source.id}><summary>{source.id} · {source.title} / {source.heading}</summary><p>{source.excerpt}</p><small>{source.owner} · Updated {source.updated}</small></details>)}</div>}
            </div>
          </article>)}</div>
          {busy && <div className="thinking" role="status">Searching company knowledge and preparing an answer…</div>}
          <div ref={endRef} />
        </section>
        <div className="composer-wrap">
          {error && <p className="error" role="alert">{error}</p>}
          <form className="composer" onSubmit={onSubmit}>
            <label htmlFor="question" className="sr-only">Your question</label>
            <input id="question" value={question} onChange={event => setQuestion(event.target.value)} placeholder="Ask about a policy, process, or procedure..." maxLength={2000} disabled={busy} />
            <button type="submit" disabled={busy || !question.trim()} aria-label="Send question">↑</button>
          </form>
          <small>Responses are generated by AI. Check cited passages for critical decisions.</small>
        </div>
      </main>
    </div>
  )
}
