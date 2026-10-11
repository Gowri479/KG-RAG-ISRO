# Phase 1 corpus audit — baseline

**Audit date:** 2026-10-07  
**Scope:** Existing repository state, before Phase 1 corpus additions.

## Existing pilot resources

- The repository has an Aditya-L1-only `data/source_manifest.csv` with 7 records: 5 local official-source files and 2 PRADAN pages recorded as failed with HTTP 504. The 5 local files are 3 PDFs and 2 HTML pages in `data/documents/aditya_l1/official/`.
- The local documents are the Aditya-L1 mission booklet, payload document, ALPPS user guide, ISSDC mission page, and ALPPS portal landing page. The existing manifest records source URLs, extraction/page counts, authority notes, and some extraction status, but is not a general document registry.
- `data/raw/` contains 845 Markdown page extracts (about 402 MiB) covering 791 filename-level source families; `data/chunks/` contains 848 JSON files (about 1.33 GiB). `data/kg/` contains the pilot graph, pickle, chunk map, and community summaries. The graph JSON contains 31,358 nodes and 95,671 edges.
- Existing preprocessing in `src/preprocessing/clean.py` deduplicates lines and removes broad text patterns; `src/preprocessing/chunk.py` creates fixed word windows. These are pilot utilities, not section- and page-preserving corpus construction.
- Existing KG code uses generic NER/dependency heuristics and has a broader, partly uncontrolled relation map. A baseline check of `data/kg/knowledge_graph.json` found 482 of 95,671 edges with `document_id` and `source_url` fields, and no `chunk_id` on the checked edge records. Nodes are serialized as untyped strings. Thus the existing KG is not a source of fully supported Phase 1 triples.
- `src/scraper/crawl.py` is the existing scraper. Its outputs and the generated raw/chunk/KG artifacts are pilot data; they will not be overwritten by Phase 1.
- `urls.txt` has 58 seed URLs, including Wikipedia and other non-primary URLs. Those sources are not evidence for the Phase 1 KG; only inspected primary/authoritative sources are candidates for the new registry.
- Existing benchmark datasets, checkpoints, experiment reports/results, and `paper/main.tex` are present. They are out of scope and will not be edited, regenerated, or used to select corpus sources.

## Baseline gaps against Phase 1

1. The current registry/corpus is concentrated on one mission and does not establish cross-mission, launch, institutional, temporal, and science coverage.
2. Current chunks do not provide the required complete document → section/page → chunk lineage.
3. Existing KG typing, relation control, and evidence provenance are insufficient for the research question; the pilot graph must remain unchanged and separate from the new KG-ready records.
4. There is no dedicated ontology, validated relation schema, duplicate-hash check, corpus-wide structural validator, or reproducible coverage/gap report.
5. Existing extraction includes an image-only payload PDF and unreliable/failed portal retrievals; these limitations must remain explicit rather than being silently treated as covered evidence.

## Preservation and collection policy

No pilot benchmark, result, paper, index, raw extraction, chunk set, or graph was modified during this audit. Phase 1 will add a separate registry, ontology, relation vocabulary, cleaned corpus, semantic chunks, and evidence-linked KG-ready records. Existing source files will be reused where appropriate rather than copied. Candidate sources will be assessed from the official document/page itself, with authority and access status recorded; no Tier 4 source will be used to establish facts where a primary source is available.
