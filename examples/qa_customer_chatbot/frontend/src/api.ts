export type ChatResponse = {
  conversation_id: string;
  message_id: string;
  answer: string;
  order?: unknown;
  sources?: unknown;
  guardrail?: unknown;
};

export type FeedbackResponse = Record<string, unknown>;
export type Rating = "up" | "down";

const baseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options?.headers },
    });
  } catch {
    throw new Error("Could not reach the service. Check your connection and try again.");
  }

  if (!response.ok) {
    let detail: unknown;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = body.detail;
      }
    } catch {
      // Not all errors have a JSON body.
    }
    throw new Error(typeof detail === "string" && detail
      ? detail
      : `Request failed (${response.status}). Please try again.`);
  }
  return response.json() as Promise<T>;
}

export async function checkHealth(): Promise<void> {
  const health = await request<{ status: string; model_configured: boolean }>("/api/health");
  if (health.status !== "ok" || !health.model_configured) {
    throw new Error("The service needs a configured model provider.");
  }
}

export async function sendMessage(
  message: string,
  conversationId?: string,
  customerId?: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      ...(conversationId ? { conversation_id: conversationId } : {}),
      ...(customerId ? { customer_id: customerId } : {}),
    }),
  });
}

export async function sendFeedback(
  messageId: string,
  rating?: Rating,
  correction?: string,
): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({
      message_id: messageId,
      ...(rating ? { rating } : {}),
      ...(correction ? { correction } : {}),
    }),
  });
}
