# RCA Generator frontend

Run `npm install` and `npm run dev` from this directory. `npm run build` type-checks and produces a static site in `dist/`. The default API base is `/api`; Vite proxies it to `http://localhost:8000` during development. For a different backend, set `VITE_API_BASE_URL` to the complete API prefix (for example `http://localhost:9000/api`). The deployed server must proxy `/api` itself when using the default base.

## API contract

- `GET /api/health` → JSON with `status: "ok" | "unconfigured"`; other responses mark the API offline.
- `GET /api/incidents` → `Incident[]` or `{ "incidents": Incident[] }`. Each incident needs `id` and `title`; optional fields: `description`, `severity`, `service`, `status`, `started_at`.
- `POST /api/analyze` with `{ "incident_id": "<id>" }` → `RcaReport`, `{ "report": RcaReport }`, or `{ "analysis": RcaReport }`.

`RcaReport` accepts `summary`, `root_cause` (text or `{ description, confidence, citations }`), `confidence` (fraction, percent, or label), `impact`, `contributing_factors` (text or evidence objects), `evidence` (`{ claim | summary | description, confidence?, source?, citations?, citation_ids? }[]`), `citations` (`{ id?, source?, title?, url?, excerpt?, timestamp? }[]`), `timeline` (`{ timestamp?, title?, event?, description?, citations? }[]`), `recommendations` (text or `{ title?, action?, description?, priority? }[]`), and `generated_at`. Citation references may be citation objects or IDs matching `citations[].id`. Unavailable optional sections are hidden; no fabricated findings are displayed.
