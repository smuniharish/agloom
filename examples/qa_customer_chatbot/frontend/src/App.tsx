import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { checkHealth, sendFeedback, sendMessage, type ChatResponse, type Rating } from "./api";

type Message =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; response: ChatResponse };

type Source = { title: string; url?: string; description?: string };

const suggestions = [
  "Where is my order?",
  "What is the return policy?",
  "Can I update my delivery address?",
];

function Icon({ name, size = 20 }: { name: "spark" | "send" | "plus" | "heart" | "arrow" | "check" | "refresh"; size?: number }) {
  const paths: Record<typeof name, ReactNode> = {
    spark: <><path d="m12 2 1.7 6.3L20 10l-6.3 1.7L12 18l-1.7-6.3L4 10l6.3-1.7L12 2Z" /><path d="m19 17 .7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z" /></>,
    send: <><path d="m3 11 18-8-8 18-2.5-7.5L3 11Z" /><path d="M10.5 13.5 21 3" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    heart: <path d="M20.8 8.6c0 4.4-8.8 10-8.8 10s-8.8-5.6-8.8-10a4.7 4.7 0 0 1 8.8-2.2 4.7 4.7 0 0 1 8.8 2.2Z" />,
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    refresh: <><path d="M20 11a8 8 0 0 0-14.3-4.9L3 9" /><path d="M3 4v5h5M4 13a8 8 0 0 0 14.3 4.9L21 15" /><path d="M21 20v-5h-5" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function textOf(value: unknown): string | undefined {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return undefined;
}

function fieldLabel(key: string): string {
  return key.replace(/[_-]/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function safeUrl(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : undefined;
  } catch {
    return undefined;
  }
}

function asSources(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];
  return value.map((item, index) => {
    if (typeof item === "string") return { title: item };
    if (item && typeof item === "object") {
      const source = item as Record<string, unknown>;
      return {
        title: textOf(source.title) ?? textOf(source.name) ?? textOf(source.policy) ?? `Source ${index + 1}`,
        url: safeUrl(source.url) ?? safeUrl(source.link),
        description: textOf(source.excerpt) ?? textOf(source.snippet) ?? textOf(source.description),
      };
    }
    return { title: `Source ${index + 1}` };
  });
}

function DetailCard({ title, value, className = "" }: { title: string; value: unknown; className?: string }) {
  if (value == null || value === "") return null;
  if (typeof value === "string") {
    return <section className={`detail-card ${className}`} aria-label={title}><h4>{title}</h4><p>{value}</p></section>;
  }
  if (typeof value !== "object") {
    return <section className={`detail-card ${className}`} aria-label={title}><h4>{title}</h4><p>{String(value)}</p></section>;
  }
  const fields = Object.entries(value).filter(([, entry]) => entry != null && entry !== "").map(([key, entry]) => ({
    label: fieldLabel(key),
    value: textOf(entry) ?? (Array.isArray(entry) ? entry.map(textOf).filter(Boolean).join(", ") : undefined),
  })).filter((field) => field.value);
  if (!fields.length) return null;
  return (
    <section className={`detail-card ${className}`} aria-label={title}>
      <h4>{title}</h4>
      <dl>{fields.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    </section>
  );
}

function Feedback({ messageId }: { messageId: string }) {
  const [rating, setRating] = useState<Rating>();
  const [correction, setCorrection] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  async function submit(nextRating?: Rating, nextCorrection?: string) {
    setPending(true);
    setError("");
    setNotice("");
    try {
      await sendFeedback(messageId, nextRating, nextCorrection);
      setRating(nextRating);
      if (nextCorrection) {
        setIsEditing(false);
        setCorrection("");
        setNotice("Correction received. Thank you for helping us improve.");
      } else {
        setNotice("Thanks for your feedback.");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save feedback.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="feedback">
      <div className="feedback-actions" aria-label="Rate this answer">
        <span>Was this helpful?</span>
        <button type="button" className={`rating-button ${rating === "up" ? "selected" : ""}`} aria-label="Helpful answer" aria-pressed={rating === "up"} disabled={pending} onClick={() => void submit("up")} title="Helpful answer">👍</button>
        <button type="button" className={`rating-button ${rating === "down" ? "selected" : ""}`} aria-label="Not helpful answer" aria-pressed={rating === "down"} disabled={pending} onClick={() => void submit("down")} title="Not helpful answer">👎</button>
        <button type="button" className="correction-toggle" aria-expanded={isEditing} disabled={pending} onClick={() => { setIsEditing(!isEditing); setError(""); }}>
          {isEditing ? "Cancel" : "Suggest a correction"}
        </button>
      </div>
      {isEditing && (
        <form className="correction-form" onSubmit={(event) => { event.preventDefault(); if (correction.trim()) void submit(rating, correction.trim()); }}>
          <label htmlFor={`correction-${messageId}`}>What should we have said instead?</label>
          <textarea id={`correction-${messageId}`} value={correction} onChange={(event) => setCorrection(event.target.value)} placeholder="Share the correct information…" rows={3} maxLength={2000} required disabled={pending} />
          <button type="submit" className="small-primary" disabled={pending || !correction.trim()}>{pending ? "Sending…" : "Send correction"}</button>
        </form>
      )}
      {notice && <p className="feedback-notice" role="status"><Icon name="check" size={15} />{notice}</p>}
      {error && <p className="feedback-error" role="alert">{error}</p>}
    </div>
  );
}

function AssistantMessage({ response }: { response: ChatResponse }) {
  const sources = asSources(response.sources);
  return (
    <div className="message-row assistant-row">
      <div className="assistant-avatar" aria-hidden="true"><Icon name="spark" size={19} /></div>
      <div className="assistant-content">
        <div className="message-meta">CARE ASSISTANT <span>·</span> JUST NOW</div>
        <div className="answer">{response.answer}</div>
        <div className="details-grid">
          <DetailCard title="Order details" value={response.order} className="order-card" />
          <DetailCard title="Policy & guidance" value={response.guardrail} className="guardrail-card" />
        </div>
        {sources.length > 0 && (
          <section className="sources" aria-label="Sources and citations">
            <h4>Sources & references</h4>
            <ul>{sources.map((source, index) => <li key={`${source.title}-${index}`}>
              <span className="source-number">{String(index + 1).padStart(2, "0")}</span>
              <span className="source-body">
                {source.url ? <a href={source.url} target="_blank" rel="noopener noreferrer">{source.title} <span aria-hidden="true">↗</span></a> : <span className="source-title">{source.title}</span>}
                {source.description && <span className="source-description">{source.description}</span>}
              </span>
            </li>)}</ul>
          </section>
        )}
        {response.message_id && <Feedback messageId={response.message_id} />}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string>();
  const [customerId, setCustomerId] = useState("");
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const transcript = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let active = true;
    async function updateHealth() {
      try {
        await checkHealth();
        if (active) setHealth("online");
      } catch {
        if (active) setHealth("offline");
      }
    }
    void updateHealth();
    const timer = window.setInterval(() => void updateHealth(), 30000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (transcript.current) transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [messages, sending]);

  async function submit(message: string) {
    const text = message.trim();
    if (!text || sending) return;
    setSending(true);
    setChatError("");
    setDraft("");
    const localId = crypto.randomUUID();
    setMessages((current) => [...current, { id: localId, role: "user", text }]);
    try {
      const response = await sendMessage(text, conversationId, customerId.trim() || undefined);
      setConversationId(response.conversation_id || conversationId);
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", response }]);
    } catch (cause) {
      setMessages((current) => current.filter((item) => item.id !== localId));
      setDraft(text);
      setChatError(cause instanceof Error ? cause.message : "Something went wrong. Please try again.");
    } finally {
      setSending(false);
      input.current?.focus();
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submit(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submit(draft);
    }
  }

  function newConversation() {
    setMessages([]);
    setConversationId(undefined);
    setDraft("");
    setChatError("");
    input.current?.focus();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-icon"><Icon name="spark" size={25} /></span><span><strong>goodcompany</strong><small>CUSTOMER CARE</small></span></div>
        <div className="sidebar-body">
          <div className="sidebar-label">YOUR WORKSPACE</div>
          <div className="sidebar-item active"><span className="sidebar-item-icon"><Icon name="heart" size={19} /></span> Support chat <span className="active-dot" /></div>
          <button type="button" className="new-chat" onClick={newConversation} disabled={sending}><Icon name="plus" size={18} /> New conversation</button>
          <div className="sidebar-divider" />
          <div className="sidebar-label">PERSONALIZE YOUR CHAT</div>
          <label className="customer-label" htmlFor="customer-id">Customer ID <span>optional</span></label>
          <input id="customer-id" className="customer-input" type="text" autoComplete="off" value={customerId} onChange={(event) => setCustomerId(event.target.value)} placeholder="e.g. CUST-1024" disabled={sending || messages.length > 0} />
          <p className="customer-hint">Add an ID before chatting to get help with your orders. Start a new conversation to change it.</p>
        </div>
        <div className="sidebar-footer"><span className="footer-orb"><Icon name="spark" size={20} /></span><div><strong>Here to help.</strong><p>Answers grounded in your orders and policies.</p></div></div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-icon"><Icon name="spark" size={20} /></span><strong>goodcompany</strong></div>
          <div className="breadcrumb"><span>Customer care</span><span className="breadcrumb-slash">/</span><strong>Support chat</strong></div>
          <div className="health-group">
            <span className={`health-pill ${health}`} role="status" aria-label={`Service ${health}`}><span className="health-dot" />{health === "online" ? "Service online" : health === "offline" ? "Service unavailable" : "Checking service"}</span>
            <button type="button" className="health-refresh" aria-label="Check service status again" title="Check service status again" onClick={() => { setHealth("checking"); void checkHealth().then(() => setHealth("online")).catch(() => setHealth("offline")); }}><Icon name="refresh" size={17} /></button>
          </div>
        </header>

        <div className="chat-page">
          <div className="chat-heading"><span className="eyebrow"><span className="eyebrow-line" /> SUPPORT, MADE SIMPLE</span><h1>How can we help you today<span className="heading-accent">?</span></h1><p>Ask us anything about orders, delivery, returns, or our policies.</p></div>
          <div className="chat-card">
            <div className="chat-card-header"><div className="chat-card-title"><span className="chat-header-icon"><Icon name="spark" size={19} /></span><div><strong>Care assistant</strong><span>Your personal support companion</span></div></div><span className="private-badge"><span /> A fresh conversation</span></div>
            <div className="transcript" ref={transcript} role="log" aria-label="Conversation" aria-live="polite" aria-relevant="additions">
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-icon"><Icon name="spark" size={31} /></div>
                  <span className="welcome-kicker">HELLO THERE</span>
                  <h2>Great support starts here.</h2>
                  <p>I'm here to help you find answers, check an order, and make your day a little easier. What’s on your mind?</p>
                  <div className="suggestions" aria-label="Suggested questions">{suggestions.map((suggestion) => <button type="button" key={suggestion} disabled={sending} onClick={() => void submit(suggestion)}>{suggestion}<Icon name="arrow" size={16} /></button>)}</div>
                </div>
              ) : (
                <div className="message-list">{messages.map((item) => item.role === "user"
                  ? <div className="message-row user-row" key={item.id}><div className="user-content"><div className="message-meta">YOU <span>·</span> JUST NOW</div><div className="user-bubble">{item.text}</div></div><span className="user-avatar" aria-hidden="true">Y</span></div>
                  : <AssistantMessage response={item.response} key={item.id} />)}
                  {sending && <div className="message-row assistant-row"><div className="assistant-avatar" aria-hidden="true"><Icon name="spark" size={19} /></div><div className="typing" role="status" aria-label="Assistant is responding"><span /><span /><span /></div></div>}
                </div>
              )}
            </div>
            <div className="composer-area">
              {chatError && <p className="chat-error" role="alert">{chatError}</p>}
              <form className="composer" onSubmit={onSubmit}>
                <label htmlFor="message-input" className="sr-only">Your message</label>
                <textarea id="message-input" ref={input} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={onKeyDown} placeholder="Type your question here…" rows={2} maxLength={4000} disabled={sending} />
                <button type="submit" className="send-button" aria-label="Send message" disabled={sending || !draft.trim()}><Icon name="send" size={19} /></button>
              </form>
              <div className="composer-caption"><span>Press <kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line</span><span>Thoughtful answers, every time <span className="caption-spark">✦</span></span></div>
            </div>
          </div>
          <p className="bottom-note">Answers may need verification. For urgent matters, please contact our support team directly.</p>
        </div>
      </main>
    </div>
  );
}
