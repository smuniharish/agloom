# QA customer chatbot frontend

React, TypeScript and Vite UI for the customer-support example. From this directory run `npm.cmd install`, then `npm.cmd run dev` or `npm.cmd run build` in PowerShell. Open the URL printed by Vite.

The default API path is same-origin `/api`; Vite proxies it to `http://localhost:8000` during development. To use a different backend, copy `.env.example` to `.env.local` and set `VITE_API_BASE_URL` to its origin (without `/api`). The backend must allow cross-origin requests when using a separate origin. Only `VITE_`-prefixed, public configuration belongs here; do not put secrets in frontend environment files.

The interface checks `GET /api/health` and shows the service as unavailable when its model is unconfigured. It sends `POST /api/chat` with `message` and optional `conversation_id`/`customer_id`, and sends `POST /api/feedback` with `message_id` and a rating (`up` or `down`) and/or a correction. The chat response includes `conversation_id`, `message_id`, `answer`, a verified `order` or null, `sources`, and `guardrail`. Conversation and feedback state remain in memory and reset on page reload.
