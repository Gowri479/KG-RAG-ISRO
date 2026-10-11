// TypeScript types for the KG-RAG ISRO API (src/api/schemas.py).
// The machine-readable spec is openapi.json in this folder.

export type Mode = "live" | "mock";

export interface AskRequest {
  question: string; // 3-500 characters
}

/** Official document a KG fact was curated from. */
export interface Source {
  document_id: string;
  url: string | null;
  section: string | null;
  page: string | null;
  excerpt: string | null; // supporting text from the official source
}

/** One fact in a KG path: subject -> relation -> object. */
export interface Hop {
  subject: string;
  relation: string; // e.g. "HAS_PAYLOAD", "DEVELOPED_BY"
  object: string;
  triple_id: string;
  source: Source;
}

export interface KGPath {
  path_id: string;
  score: number; // 0-1, paths arrive sorted by score
  hops: Hop[]; // 1 hop = direct fact, 2 hops = chained facts
}

/** Text passage from dense retrieval (empty in mock mode). */
export interface Passage {
  rank: number;
  text: string;
  url: string | null;
  title: string | null;
  mission: string | null;
  source_file: string | null;
}

export interface QueryAnalysis {
  query_type: "FACTUAL" | "RELATIONAL" | "MULTI_HOP";
  entities: string[]; // e.g. "mission:aditya-l1"
  relations: string[];
  hop_depth: number;
}

export interface Usage {
  latency_ms: number;
  prompt_tokens: number | null;
  context_tokens_used: number | null;
  context_trimmed: boolean | null;
}

export interface AskResponse {
  question: string;
  answer: string;
  abstained: boolean; // true when the answer is "I don't know."
  mode: Mode; // "mock" answers are canned: never present them as real
  system: string;
  analysis: QueryAnalysis;
  kg_paths: KGPath[]; // top 10 for display
  kg_paths_total: number; // all paths given to the model
  passages: Passage[];
  usage: Usage;
}

export interface Example {
  question: string;
  mission: string;
  category: string;
}

export interface ExamplesResponse {
  examples: Example[];
}

export interface HealthResponse {
  status: "ready" | "warming_up" | "error";
  mode: Mode;
  model: string;
  ollama_reachable: boolean | null;
  detail: string | null;
}

/** Body of 422 (invalid input) and 503 (not ready / model unavailable) responses. */
export interface ErrorResponse {
  detail: string | unknown[];
}
