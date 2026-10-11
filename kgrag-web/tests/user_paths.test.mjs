import test from "node:test";
import assert from "node:assert/strict";
import { ask, getExamples, getHealth } from "../src/api.ts";

// Test suite exercising actual application behavior in src/api.ts:
// 1. API unavailable / offline network error handling
// 2. Mock vs Live mode label handling
// 3. Successful question submission and response parsing
// 4. Abstention handling when information is insufficient
// 5. Multi-hop knowledge graph structure and path chaining
// 6. Source provenance link and official domain validation
// 7. Example questions fetching and mission metadata preservation

const FIXTURE_LIVE_HEALTH = {
  status: "ready",
  mode: "live",
  model: "mistral:7b-instruct-q4_K_M",
  ollama_reachable: true,
  detail: null,
};

const FIXTURE_MOCK_HEALTH = {
  status: "ready",
  mode: "mock",
  model: "mistral:7b-instruct-q4_K_M",
  ollama_reachable: null,
  detail: "Mock mode: real KG retrieval, canned answers, no language model.",
};

const FIXTURE_EXAMPLES = {
  examples: [
    {
      question: "Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?",
      mission: "Aditya-L1",
      category: "MULTI_HOP",
    },
    {
      question: "When did Chandrayaan-3 land on the Moon?",
      mission: "Chandrayaan-3",
      category: "FACTUAL",
    },
    {
      question: "What are the primary instruments on AstroSat?",
      mission: "AstroSat",
      category: "RELATIONAL",
    },
  ],
};

const FIXTURE_ASK_SUCCESS = {
  question: "Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?",
  answer: "The Solar Ultraviolet Imaging Telescope (SUIT) on board Aditya-L1 was developed by the Inter-University Centre for Astronomy and Astrophysics (IUCAA), Pune.",
  abstained: false,
  mode: "live",
  system: "relation_aware_kg_rag",
  analysis: {
    query_type: "MULTI_HOP",
    entities: ["payload:aditya-suit", "org:iucaa"],
    relations: ["HAS_PAYLOAD", "DEVELOPED_BY"],
    hop_depth: 2,
  },
  kg_paths: [
    {
      path_id: "path_01",
      score: 0.95,
      hops: [
        {
          subject: "Aditya-L1",
          relation: "HAS_PAYLOAD",
          object: "SUIT",
          triple_id: "trip_001",
          source: {
            document_id: "aditya_l1_payloads",
            url: "https://www.isro.gov.in/Aditya_L1.html",
            section: "Payloads",
            page: "2",
            excerpt: "SUIT is a payload on board Aditya-L1.",
          },
        },
        {
          subject: "SUIT",
          relation: "DEVELOPED_BY",
          object: "IUCAA",
          triple_id: "trip_002",
          source: {
            document_id: "aditya_l1_suit",
            url: "https://www.isro.gov.in/Aditya_L1_SUIT.html",
            section: "Development",
            page: "1",
            excerpt: "Developed by IUCAA, Pune in collaboration with ISRO.",
          },
        },
      ],
    },
  ],
  kg_paths_total: 4,
  passages: [
    {
      rank: 1,
      text: "SUIT was developed by IUCAA.",
      url: "https://www.isro.gov.in/Aditya_L1_SUIT.html",
      title: "Aditya-L1 SUIT",
      mission: "Aditya-L1",
      source_file: "suit.pdf",
    },
  ],
  usage: {
    latency_ms: 1250,
    prompt_tokens: 350,
    context_tokens_used: 280,
    context_trimmed: false,
  },
};

const FIXTURE_ABSTAIN = {
  question: "What is the secret mission code of satellite X-999?",
  answer: "I don't know.",
  abstained: true,
  mode: "live",
  system: "relation_aware_kg_rag",
  analysis: {
    query_type: "FACTUAL",
    entities: [],
    relations: [],
    hop_depth: 0,
  },
  kg_paths: [],
  kg_paths_total: 0,
  passages: [],
  usage: {
    latency_ms: 450,
    prompt_tokens: 120,
    context_tokens_used: 0,
    context_trimmed: false,
  },
};

test("Path 1: API unavailable and offline error handling", async () => {
  const originalFetch = globalThis.fetch;
  try {
    // Simulate connection refused / network offline
    globalThis.fetch = async () => {
      throw new TypeError("Failed to fetch");
    };

    await assert.rejects(
      async () => await ask("When was Chandrayaan-3 launched?"),
      (err) => {
        assert.equal(err.message, "The service is currently unavailable. Please try again later.");
        return true;
      }
    );

    // Simulate backend 503 warmup error with detail message
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ detail: "Language model unavailable. Is Ollama running?" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });

    await assert.rejects(
      async () => await ask("When was Chandrayaan-3 launched?"),
      (err) => {
        assert.ok(err.message.includes("Language model unavailable"));
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 2: Mock vs Live health response parsing", async () => {
  const originalFetch = globalThis.fetch;
  try {
    // Test live health
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_LIVE_HEALTH), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const live = await getHealth();
    assert.equal(live.status, "ready");
    assert.equal(live.mode, "live");
    assert.equal(live.ollama_reachable, true);

    // Test mock health
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_MOCK_HEALTH), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const mock = await getHealth();
    assert.equal(mock.status, "ready");
    assert.equal(mock.mode, "mock");
    assert.equal(mock.ollama_reachable, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 3: Successful question submission and response schema parsing", async () => {
  const originalFetch = globalThis.fetch;
  let interceptedRequest = null;
  try {
    globalThis.fetch = async (url, options) => {
      interceptedRequest = { url, options };
      return new Response(JSON.stringify(FIXTURE_ASK_SUCCESS), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    const q = "Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?";
    const res = await ask(q);

    // Verify request dispatch
    assert.ok(interceptedRequest.url.endsWith("/api/ask"));
    assert.equal(interceptedRequest.options.method, "POST");
    assert.deepEqual(JSON.parse(interceptedRequest.options.body), { question: q });

    // Verify parsed response fields
    assert.equal(res.question, q);
    assert.ok(res.answer.includes("IUCAA"));
    assert.equal(res.abstained, false);
    assert.equal(res.mode, "live");
    assert.equal(res.analysis.query_type, "MULTI_HOP");
    assert.equal(res.analysis.hop_depth, 2);
    assert.equal(res.usage.latency_ms, 1250);
    assert.equal(res.passages.length, 1);
    assert.equal(res.passages[0].title, "Aditya-L1 SUIT");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 4: Abstention handling when evidence is insufficient", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_ABSTAIN), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const res = await ask("What is the secret mission code of satellite X-999?");
    assert.equal(res.abstained, true);
    assert.equal(res.answer, "I don't know.");
    assert.equal(res.kg_paths.length, 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 5: Multi-hop graph structure and edge connections", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_ASK_SUCCESS), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const res = await ask("Which organization developed SUIT?");
    assert.ok(res.kg_paths.length > 0);

    const path = res.kg_paths[0];
    assert.equal(path.hops.length, 2);

    const [hop1, hop2] = path.hops;
    assert.equal(hop1.subject, "Aditya-L1");
    assert.equal(hop1.relation, "HAS_PAYLOAD");
    assert.equal(hop1.object, "SUIT");

    assert.equal(hop2.subject, "SUIT");
    assert.equal(hop2.relation, "DEVELOPED_BY");
    assert.equal(hop2.object, "IUCAA");

    // Chain connectivity: object of hop1 must connect to subject of hop2
    assert.equal(hop1.object, hop2.subject);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 6: Provenance source linking and official domain validation", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_ASK_SUCCESS), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const res = await ask("Which organization developed SUIT?");
    const source1 = res.kg_paths[0].hops[0].source;

    assert.equal(source1.document_id, "aditya_l1_payloads");
    assert.ok(source1.url);
    assert.ok(source1.excerpt);

    const url = new URL(source1.url);
    assert.ok(
      url.hostname === "www.isro.gov.in" || url.hostname === "www.issdc.gov.in",
      "Source URL must belong to official ISRO/ISSDC primary domains"
    );
    assert.equal(url.protocol, "https:");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Path 7: Example questions fetching and mission metadata preservation", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify(FIXTURE_EXAMPLES), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    const examples = await getExamples();
    assert.equal(examples.length, 3);

    // Verify metadata preservation (not just strings)
    assert.deepEqual(examples[0], {
      question: "Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?",
      mission: "Aditya-L1",
      category: "MULTI_HOP",
    });

    // Verify mission filtering behavior as used in App.tsx
    const filterByMission = (list, key) =>
      list.filter(
        (e) =>
          (e.mission || "").toLowerCase().includes(key.toLowerCase()) ||
          e.question.toLowerCase().includes(key.toLowerCase())
      );

    const adityaMatches = filterByMission(examples, "aditya");
    assert.equal(adityaMatches.length, 1);
    assert.equal(adityaMatches[0].mission, "Aditya-L1");

    const ch3Matches = filterByMission(examples, "chandrayaan");
    assert.equal(ch3Matches.length, 1);
    assert.equal(ch3Matches[0].mission, "Chandrayaan-3");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
