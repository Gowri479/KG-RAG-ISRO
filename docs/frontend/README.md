# Frontend guide: KG-RAG ISRO API

This folder has everything needed to build a frontend for the KG-RAG system
in any framework (React, Vue, Svelte, plain JS). The frontend talks to a
local HTTP API and never touches the model, the data or the Python code.

| File | What it is |
|---|---|
| `README.md` | This guide |
| `api-types.ts` | TypeScript types for every request and response |
| `openapi.json` | Machine-readable API spec (can generate a typed client) |
| `example_ask_response.json` | A real `/api/ask` response |

## What the system does

A user asks a question about an ISRO mission (Aditya-L1, AstroSat,
Chandrayaan-1/2/3, Gaganyaan, Mars Orbiter Mission). The system finds the
relevant facts in a **knowledge graph** built from official ISRO/ISSDC
documents, adds supporting text passages, and a local language model
(Mistral-7B) writes the answer.

The main selling point is **provenance**. Every answer comes with:
- the KG facts it used (e.g. `Aditya-L1 → HAS_PAYLOAD → SUIT → DEVELOPED_BY → IUCAA`);
- for each fact, the official document, its URL and the supporting excerpt.

The UI should make these visible, not hide them.

## Running the API

You need Python 3.11 and the repository. From the repository root:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Mock mode (start here)

No GPU, no Ollama, no large data files. KG facts and sources are real;
answers are canned. Benchmark questions return the system's real recorded
answer, and other questions return a reply marked `[MOCK ANSWER]`.

```bash
python -m src.api --mock
```

### Live mode (real answers)

Needs Ollama running with `mistral:7b-instruct-q4_K_M`, plus the local files
`data/index/faiss_index.index` and `data/chunks/chunks.json`, which are not in
git (ask Nandana for them). Start-up takes about 2 minutes; answers take
roughly 5-30 seconds.

```bash
python -m src.api
```

Both modes serve at **http://127.0.0.1:8000**, with interactive docs at
**http://127.0.0.1:8000/docs**.

CORS allows `http://localhost:5173` (Vite) and `http://localhost:3000` by
default. For another dev server, set `KG_RAG_CORS_ORIGINS`
(comma-separated) before starting. `KG_RAG_MOCK_DELAY_MS` (default 800)
simulates latency in mock mode.

## Endpoints

### `GET /api/health`

```json
{ "status": "ready", "mode": "live", "model": "mistral:7b-instruct-q4_K_M", "ollama_reachable": true, "detail": null }
```

`status` is `warming_up` → `ready`, or `error` (reason in `detail`). Poll
every 2-3 seconds on page load and keep the Ask button disabled until
`ready`.

### `GET /api/examples`

51 example questions, each `{ question, mission, category }`. Good for
suggestion chips or a "try asking" list.

### `POST /api/ask`

Request:

```json
{ "question": "Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?" }
```

Response (see `example_ask_response.json` for a full one):

| Field | Use in the UI |
|---|---|
| `answer` | The main answer text |
| `abstained` | `true` when the answer is "I don't know." Show a friendly "no answer found in official sources" state |
| `mode` | If `"mock"`, show a visible "demo data" badge |
| `kg_paths[]` | Facts behind the answer, sorted by `score`. Each has `hops[]` of `subject → relation → object`, and each hop has a `source` with `document_id`, `url` and `excerpt` |
| `kg_paths_total` | For "showing 10 of N facts" |
| `passages[]` | Supporting text passages (`text`, `url`). Empty in mock mode |
| `analysis` | Detected entities, relations and `query_type` (`FACTUAL`, `RELATIONAL`, `MULTI_HOP`). Nice for a "how it understood your question" panel |
| `usage.latency_ms` | Response time |

Errors:

| Status | When | `detail` |
|---|---|---|
| 422 | Question empty or over 500 characters | validation list |
| 503 | Still warming up, or the model (Ollama) is unavailable | message string |

## Minimal client

```ts
import type { AskResponse, HealthResponse } from "./api-types";

const API = "http://127.0.0.1:8000";

export async function ask(question: string): Promise<AskResponse> {
  const res = await fetch(`${API}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res.json();
}

export async function health(): Promise<HealthResponse> {
  return (await fetch(`${API}/api/health`)).json();
}
```

## UI notes

- **Loading:** live answers take up to ~30 seconds. Show progress (e.g.
  "Searching the knowledge graph… Writing the answer…") and disable re-submit.
  Requests are handled one at a time.
- **Provenance view:** render each KG path as a chain of chips:
  `Aditya-L1` → *has payload* → `SUIT` → *developed by* → `IUCAA`.
  Clicking a hop shows its source excerpt and links to the official URL.
  Paths with 2 hops are the interesting multi-hop reasoning; consider
  highlighting them.
- **Relation labels:** convert `HAS_PAYLOAD` to "has payload" (lowercase,
  underscores to spaces).
- **Abstention is a feature:** "I don't know" means the system refused to
  guess. Present it neutrally.
- **Mock badge:** when `mode === "mock"`, show it clearly. Mock answers must
  never be presented or demoed as real.

## Regenerating these files

If the API changes:

```bash
python -m src.api.export_docs
```

Then update `api-types.ts` to match `src/api/schemas.py`.
