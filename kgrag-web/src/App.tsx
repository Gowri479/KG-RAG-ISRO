import { Fragment, useEffect, useState } from "react";
import { ask, getExamples, getHealth } from "./api";
import type { AskResponse, Health, Hop } from "./api";
import "./App.css";

const MISSIONS = [
  { name: "Aditya-L1", tag: "Solar mission", key: "aditya" },
  { name: "Chandrayaan", tag: "Lunar mission", key: "chandrayaan" },
  { name: "AstroSat", tag: "Space observatory", key: "astrosat" },
  { name: "Gaganyaan", tag: "Human spaceflight", key: "gaganyaan" },
  { name: "Mars Orbiter", tag: "Planetary mission", key: "mars" },
];

const ROLES: Record<string, [string, string]> = {
  HAS_PAYLOAD: ["Mission", "Payload"],
  DEVELOPED_BY: ["Payload", "Organization"],
};
const GLYPH: Record<string, string> = { Mission: "◉", Payload: "◆", Organization: "▣", Entity: "●" };
const words = (s: string) => s.replace(/_/g, " ").toLowerCase();

function Graph({ hops, sel, onSel }: { hops: Hop[]; sel: string | null; onSel: (id: string) => void }) {
  const role = (h: Hop, i: 0 | 1) => (ROLES[h.relation] ?? ["Entity", "Entity"])[i];
  const node = (name: string, type: string, c: number) => (
    <div className={`gnode c${c % 3}`}>
      <span className="ring">{GLYPH[type] ?? "●"}</span>
      <b>{name}</b>
      <small>{type}</small>
    </div>
  );
  return (
    <div className="graph">
      {node(hops[0].subject, role(hops[0], 0), 0)}
      {hops.map((h, i) => (
        <Fragment key={h.triple_id}>
          <button
            className={"edge" + (sel === h.triple_id ? " on" : "")}
            onClick={() => onSel(h.triple_id)}
            aria-label={`Show the source for ${words(h.relation)}`}
          >
            <small>{words(h.relation)}</small>
            <span />
          </button>
          {node(h.object, role(h, 1), i + 1)}
        </Fragment>
      ))}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [mission, setMission] = useState(0);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);
  const [more, setMore] = useState(false);
  const [sel, setSel] = useState<string | null>(null);

  const ready = health?.status === "ready";

  useEffect(() => {
    let stop = false;
    const check = async () => {
      try {
        const h = await getHealth();
        if (stop) return;
        setHealth(h);
        if (h.status === "ready") {
          setExamples(await getExamples());
          return;
        }
      } catch {
        /* backend not reachable yet */
      }
      if (!stop) setTimeout(check, 3000);
    };
    check();
    return () => {
      stop = true;
    };
  }, []);

  const submit = async (q: string) => {
    if (!q.trim() || loading) return;
    setQuestion(q);
    setAsked(q);
    setLoading(true);
    setError(null);
    setResult(null);
    setSel(null);
    setMenu(false);
    try {
      setResult(await ask(q));
    } catch (e) {
      setError(e instanceof Error ? e.message : "The service is currently unavailable. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setLoading(false);
    setQuestion("");
    setMenu(false);
  };

  const key = MISSIONS[mission].key;
  const matched = examples.filter((e) => e.toLowerCase().includes(key));
  const list = matched.length ? matched : examples;
  const shown = more ? list : list.slice(0, 6);

  const seen = new Set<string>();
  const hops = (result?.kg_paths ?? [])
    .flatMap((p) => p.hops)
    .filter((h) => !seen.has(h.triple_id) && seen.add(h.triple_id));
  const docCount = new Set(hops.map((h) => h.source.document_id)).size;
  const topScore = result?.kg_paths[0]?.score;
  const traceView = loading || !!result || !!error;
  const abstained = !!result?.abstained;

  const picker = (cls: string) =>
    MISSIONS.map((m, i) => (
      <button
        key={m.name}
        className={cls + (i === mission ? " on" : "")}
        onClick={() => {
          setMission(i);
          reset();
        }}
      >
        <b>{m.name}</b>
        <small>{m.tag}</small>
      </button>
    ));

  return (
    <div className="app">
      <header className="mbar">
        <button onClick={() => setMenu(!menu)} aria-label="Open menu">☰</button>
        <b>ASTRA</b>
        <span className={"led" + (ready ? " ok" : "")} />
      </header>

      <aside className={"side" + (menu ? " open" : "")}>
        <div className="brand">
          <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
            <ellipse cx="18" cy="18" rx="16" ry="6.5" fill="none" stroke="#FF8A00" strokeWidth="1.6" transform="rotate(-28 18 18)" />
            <circle cx="18" cy="18" r="5" fill="#FF8A00" />
          </svg>
          <div>
            <b>ASTRA</b>
            <small>ISRO knowledge assistant</small>
          </div>
        </div>
        <button className="nav on" onClick={reset}>Home</button>
        <p className="grp">Mission index</p>
        {picker("mi")}
        <div className="foot">
          <span className={"led" + (ready ? " ok" : "")} />
          <div>
            <b>{ready ? "System ready" : "Connecting…"}</b>
            <small>{health?.model ? `Model: ${health.model}` : "Waiting for the API"}</small>
          </div>
        </div>
      </aside>

      <main className="main">
        {!ready ? (
          <section className="card state">
            <div>
              <h2>Waiting for the archive</h2>
              <p className="soft">
                The API isn't reachable yet. Start it with <code>python -m src.api --mock</code>. This page connects on its own.
              </p>
            </div>
          </section>
        ) : !traceView ? (
          <>
            <section className="hero">
              <div className="scene" aria-hidden="true">
                <div className="earth" />
                <svg viewBox="0 0 360 240">
                  <g transform="rotate(-18 180 120)"><ellipse className="orb o1" cx="180" cy="120" rx="160" ry="62" /></g>
                  <g transform="rotate(14 180 120)"><ellipse className="orb o2" cx="180" cy="120" rx="115" ry="40" /></g>
                  <path className="trace" d="M110 150 L200 112 L290 60" />
                  <circle className="pulse" cx="200" cy="112" r="8" />
                  <circle cx="110" cy="150" r="9" fill="#0b1e3b" stroke="#fff" strokeWidth="2" />
                  <circle cx="200" cy="112" r="9" fill="#ff8a00" stroke="#fff" strokeWidth="2" />
                  <circle cx="290" cy="60" r="9" fill="#2ea66b" stroke="#fff" strokeWidth="2" />
                  <text x="84" y="178">01 · Mission</text>
                  <text x="172" y="140">02 · Payload</text>
                  <text x="258" y="44">03 · Source</text>
                  <circle cx="40" cy="40" r="1.6" fill="#9fb6d6" />
                  <circle cx="330" cy="190" r="1.6" fill="#9fb6d6" />
                  <circle cx="60" cy="200" r="1.2" fill="#9fb6d6" />
                </svg>
              </div>
              <p className="acro">ASTRA: Aerospace Science Trace &amp; Research Assistant</p>
              <h1>
                Ask questions about <em>India's space missions.</em>
              </h1>
              <p className="lead">
                ASTRA answers from official ISRO and ISSDC documents, with a clear chain of evidence from knowledge-graph facts to the source pages.
              </p>
              <div className="composer">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submit(question)}
                  placeholder="e.g. Who developed the SUIT payload on Aditya-L1?"
                  aria-label="Your question"
                />
                <button onClick={() => submit(question)}>Ask →</button>
              </div>
            </section>

            <section className="homegrid">
              <div>
                <h2>Try these example questions</h2>
                <p className="soft sm">
                  {matched.length
                    ? `Showing questions about ${MISSIONS[mission].name}.`
                    : `Nothing mentions ${MISSIONS[mission].name} yet, so these are general.`}
                </p>
                <div className="exgrid">
                  {shown.map((ex) => (
                    <button key={ex} className="ex" onClick={() => submit(ex)}>
                      {ex}
                    </button>
                  ))}
                </div>
                {list.length > 6 && (
                  <button className="link" onClick={() => setMore(!more)}>
                    {more ? "Show fewer" : `Show all ${list.length}`}
                  </button>
                )}
              </div>
              <aside className="card rail">
                <h3>ISRO missions</h3>
                {picker("ri")}
              </aside>
            </section>
          </>
        ) : (
          <>
            <button className="link back" onClick={reset}>← Back to home</button>
            {result?.mode === "mock" && (
              <div className="banner">
                <b>Demo answer · Mock mode</b> · Not from the live system. The facts and sources below are real.
              </div>
            )}

            <section className="card qhead">
              <div>
                <small className="soft">Question</small>
                <p>{asked}</p>
              </div>
              {result && (
                <span className="pill">
                  {result.mode === "mock" ? "Mock mode" : "Live mode"}
                  {result.usage.latency_ms != null && ` · ${result.usage.latency_ms} ms`}
                </span>
              )}
            </section>

            {loading && (
              <section className="card state">
                <span className="spin" />
                <div>
                  <h3>Analyzing your question…</h3>
                  <p className="soft">This may take a few seconds.</p>
                </div>
              </section>
            )}

            {error && (
              <section className="card state bad">
                <div>
                  <h3>Something went wrong</h3>
                  <p className="soft">{error}</p>
                  <button className="outline" onClick={() => submit(asked)}>Retry</button>
                </div>
              </section>
            )}

            {result && (
              <div className="work">
                <div className="col">
                  <section className={"card answer" + (abstained ? " idk" : "")}>
                    <small className="soft">{abstained ? "I don't know" : "Answer"}</small>
                    <p>
                      {abstained
                        ? "I couldn't establish a reliable answer from the available evidence."
                        : result.answer.replace("[MOCK ANSWER] ", "")}
                    </p>
                  </section>

                  <section className="card">
                    <h3>How the question was understood</h3>
                    <dl className="kv">
                      <dt>Question type</dt>
                      <dd><span className="chip dark">{result.analysis.query_type.replace("_", "-")}</span></dd>
                      <dt>Detected entities</dt>
                      <dd>
                        {result.analysis.entities.map((e) => (
                          <span className="chip" key={e}>{e.split(":")[1] ?? e}</span>
                        ))}
                      </dd>
                      <dt>Relations</dt>
                      <dd>
                        {result.analysis.relations.map((r) => (
                          <span className="chip" key={r}>{words(r)}</span>
                        ))}
                      </dd>
                      <dt>Hop depth</dt>
                      <dd>{result.analysis.hop_depth}</dd>
                    </dl>
                  </section>

                  {!abstained && (
                    <section className="card">
                      <h3>
                        Knowledge path{" "}
                        <small className="soft">
                          {result.kg_paths.length} of {result.kg_paths_total} paths · select a relation to see its source
                        </small>
                      </h3>
                      {result.kg_paths.length === 0 && <p className="soft">No knowledge-graph facts matched this question.</p>}
                      {result.kg_paths.map((p) => (
                        <div key={p.path_id} className="path">
                          <small className="soft">Path score {p.score.toFixed(2)}</small>
                          <Graph hops={p.hops} sel={sel} onSel={(id) => setSel(sel === id ? null : id)} />
                        </div>
                      ))}
                    </section>
                  )}

                  {!abstained && result.passages.length > 0 && (
                    <section className="card">
                      <h3>Supporting text passages</h3>
                      {result.passages.map((p: any, i: number) => (
                        <details key={i} className="psg">
                          <summary>{p.title ?? p.source_file ?? `Passage ${i + 1}`}</summary>
                          <p>{p.text ?? p.content ?? p.passage ?? ""}</p>
                          {(p.source_url ?? p.url) && (
                            <a href={p.source_url ?? p.url} target="_blank" rel="noreferrer">Open source ↗</a>
                          )}
                        </details>
                      ))}
                    </section>
                  )}
                </div>

                {!abstained && (
                  <div className="col">
                    <section className="card">
                      <h3>
                        Source register <small className="soft">{hops.length} {hops.length === 1 ? "source" : "sources"}</small>
                      </h3>
                      {hops.length === 0 && <p className="soft">No source records attached.</p>}
                      {hops.map((h) => (
                        <article key={h.triple_id} className={"src" + (sel === h.triple_id ? " on" : "")}>
                          <button onClick={() => setSel(sel === h.triple_id ? null : h.triple_id)}>
                            <b>{h.source.document_id}</b>
                            <small>
                              {h.subject} → {words(h.relation)} → {h.object}
                              {h.source.page && ` · page ${h.source.page}`}
                            </small>
                          </button>
                          {sel === h.triple_id && (
                            <div className="open">
                              {h.source.section && <small className="soft">Section: {h.source.section}</small>}
                              {h.source.excerpt && <blockquote>{h.source.excerpt}</blockquote>}
                              {h.source.url && (
                                <a href={h.source.url} target="_blank" rel="noreferrer">Open official PDF ↗</a>
                              )}
                            </div>
                          )}
                        </article>
                      ))}
                    </section>

                    <section className="card">
                      <h3>Provenance summary</h3>
                      <div className="stats">
                        <div><b>{topScore != null ? topScore.toFixed(2) : "–"}</b><small>Path score</small></div>
                        <div><b>{hops.length}</b><small>Hops</small></div>
                        <div><b>{docCount}</b><small>Sources</small></div>
                      </div>
                      <p className="soft sm">
                        The path score ranks retrieved paths. It isn't the probability that the answer is correct, so check the linked sources.
                      </p>
                    </section>
                  </div>
                )}
              </div>
            )}
          </>
        )}
        <footer className="tag">
          <b>ASTRA</b> · Aerospace Science Trace &amp; Research Assistant
          <br />
          Built for researchers. Designed for truth.
        </footer>
      </main>
    </div>
  );
}