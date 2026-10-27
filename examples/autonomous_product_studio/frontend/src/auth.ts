import Keycloak from "keycloak-js";

const keycloak = new Keycloak({
  url: import.meta.env.VITE_OIDC_URL,
  realm: import.meta.env.VITE_OIDC_REALM,
  clientId: import.meta.env.VITE_OIDC_CLIENT_ID,
});

export async function initializeAuth() {
  await keycloak.init({ onLoad: "login-required", pkceMethod: "S256", checkLoginIframe: false });
  return keycloak;
}

export async function authenticatedFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  await keycloak.updateToken(30);
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${keycloak.token}`);
  return fetch(input, { ...init, headers });
}

export async function streamSse(url: string, onEvent: (value: any) => void, signal: AbortSignal) {
  let lastEventId = "";
  while (!signal.aborted) {
    await keycloak.updateToken(30);
    const headers: Record<string, string> = { Authorization: `Bearer ${keycloak.token}`, Accept: "text/event-stream" };
    if (lastEventId) headers["Last-Event-ID"] = lastEventId;
    const response = await fetch(url, { headers, signal });
    if (!response.ok || !response.body) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        if (frame.startsWith(":")) continue;
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("id:")) lastEventId = line.slice(3).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) onEvent(JSON.parse(data));
      }
    }
    if (!signal.aborted) await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

export default keycloak;
