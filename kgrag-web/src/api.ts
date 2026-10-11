const BASE = import.meta.env.VITE_API_URL;

export type Source = {
  document_id: string;
  url: string;
  section?: string;
  page?: string;
  excerpt?: string;
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
  const r = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
}