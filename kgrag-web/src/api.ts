const BASE = (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) || "http://127.0.0.1:8000";

export type Mode = "live" | "mock";

export interface Source {
  document_id: string;
  url: string | null;
  section?: string | null;
  page?: string | null;
  excerpt?: string | null;
}

export interface Hop {
  subject: string;
  relation: string;
  object: string;
  triple_id: string;
  source: Source;
}

export interface KGPath {
  path_id: string;
  score: number;
  hops: Hop[];
}

export interface Passage {
  rank: number;
  text: string;
  url?: string | null;
  title?: string | null;
  mission?: string | null;
  source_file?: string | null;
}

export interface QueryAnalysis {
  query_type: "FACTUAL" | "RELATIONAL" | "MULTI_HOP" | string;
  entities: string[];
  relations: string[];
  hop_depth: number;
}

export interface Usage {
  latency_ms: number;
  prompt_tokens?: number | null;
  context_tokens_used?: number | null;
  context_trimmed?: boolean | null;
}

export interface AskResponse {
  question: string;
  answer: string;
  abstained: boolean;
  mode: Mode;
  system: string;
  analysis: QueryAnalysis;
  kg_paths: KGPath[];
  kg_paths_total: number;
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
  ollama_reachable?: boolean | null;
  detail?: string | null;
}

export type Health = HealthResponse;

const OFFLINE = "The service is currently unavailable. Please try again later.";

async function errorMessage(r: Response): Promise<string> {
  try {
    const d = (await r.json())?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => x?.msg ?? String(x)).join("; ");
  } catch {
    /* response was not JSON */
  }
  return `Request failed (${r.status})`;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/api/health`);
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json();
}

export async function getExamples(): Promise<Example[]> {
  const r = await fetch(`${BASE}/api/examples`);
  if (!r.ok) throw new Error(await errorMessage(r));
  const data = await r.json();
  if (Array.isArray(data)) {
    return data;
  }
  if (Array.isArray(data.examples)) {
    return data.examples;
  }
  return [];
}

export async function ask(question: string): Promise<AskResponse> {
  let r: Response;
  try {
    r = await fetch(`${BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(OFFLINE);
  }
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json();
}