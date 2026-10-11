# ASTRA Web Frontend (`kgrag-web`)

**ASTRA** (Aerospace Science Trace & Research Assistant) is the canonical user-facing web interface for the ISRO Knowledge Graph RAG system. It is built with React 19, TypeScript, and Vite, connecting to the FastAPI backend (`src.api`).

---

## Features

- **Provenance-Linked Answers:** Displays responses with structured entity-relation knowledge graph paths and exact source excerpts from official ISRO/ISSDC documents.
- **Mission Index Filtering:** Explore sample questions by mission (Aditya-L1, Chandrayaan, AstroSat, Gaganyaan, Mars Orbiter) powered by API example metadata.
- **Zero-Cloud Mock Mode:** Connects to the backend in mock mode to explore real knowledge graph retrieval and canned benchmark answers with zero local GPU setup.
- **Live Inference Mode:** Connects to local 4-bit Mistral-7B via Ollama and FAISS dense vector retrieval for live question answering.

---

## Getting Started

### 1. Prerequisites

- **Node.js** >= 20.0.0
- **npm** >= 10.0.0
- Python 3.11 with project dependencies installed (`pip install -r requirements.txt`)

### 2. Start the Backend API

From the repository root, start the FastAPI server in either mock or live mode:

```bash
# Option A: Mock Mode (Recommended for quick UI testing; git-tracked files only, no Ollama needed)
python -m src.api --mock

# Option B: Live Mode (Requires Ollama running mistral:7b-instruct-q4_K_M and local FAISS index)
python -m src.api
```

The API will be available at `http://127.0.0.1:8000` with interactive OpenAPI docs at `http://127.0.0.1:8000/docs`.

### 3. Install Frontend Dependencies and Run

From the `kgrag-web` directory:

```bash
# Install dependencies reproducibly
npm ci

# Start the Vite development server
npm run dev
```

Open `http://localhost:5173` in your browser. The frontend will automatically poll `/api/health` and indicate connection status.

---

## Configuration

The frontend looks for `VITE_API_URL` to connect to the backend. By default, it targets `http://127.0.0.1:8000`. You can override this in `.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

---

## Build and Code Quality

```bash
# Run type checks and build production bundle
npm run build

# Run Oxlint linter
npm run lint

# Preview production build locally
npm run preview
```
