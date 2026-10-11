const BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export type Source = {
  document_id: string;
  url: string | null;
  section?: string | null;
  page?: string | null;
  excerpt?: string | null;
};

export type Hop = {
  subject: string;
  relation: string;
  object: string;
  triple_id: string;
  source: Source;
};

export type KGPath = {
  path_id: string;
  score: number;
  hops: Hop[];
};

export type AskResponse = {
  question: string;
  answer: string;
  abstained: boolean;
  mode: string;
  system: string;
  analysis: {
    query_type: string;
    entities: string[];
    relations: string[];
    hop_depth: number;
  };
  kg_paths: KGPath[];
  kg_paths_total: number;
  passages: any[];
  usage: { latency_ms: number | null };
};

export type Health = { status: string; mode: string; model?: string; detail?: string };

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
  return r.json();
}

export async function getExamples(): Promise<string[]> {
  const r = await fetch(`${BASE}/api/examples`);
  const data = await r.json();
  const arr = Array.isArray(data) ? data : data.examples ?? data.questions ?? [];
  return arr.map((x: any) =>
    typeof x === "string" ? x : x.question ?? x.text ?? JSON.stringify(x)
  );
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