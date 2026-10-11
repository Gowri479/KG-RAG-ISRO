# Phase 1 authoritative corpus

This directory contains the Phase 1 corpus for the research question:

> Can provenance-aware knowledge-graph augmentation improve evidence retrieval
> and factual question answering for relationship-intensive questions over
> authoritative ISRO mission documentation?

The collection is a bounded, source-audited corpus, not an attempt to scrape
every ISRO page. The existing pilot corpus, canonical benchmark, saved results,
and paper remain separate and are not inputs to source selection or fact
annotation.

## Source policy and scope

The 33 registered source records are Tier 1 sources hosted by the official
ISRO (`www.isro.gov.in`) and ISSDC (`www.issdc.gov.in`) websites. The records
cover Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Aditya-L1, Mars Orbiter
Mission, AstroSat, and Gaganyaan, with additional official launch-vehicle,
launch-archive, and ISRO-centre pages. The registry identifies every page or
document, source URL, authority tier, local path, retrieval date, status, and
SHA-256 checksum.

The image-only Chandrayaan-1 brochure remains registered and stored as a raw
source, but is excluded from text chunks because no text layer was extractable.
One Chandrayaan-2 brochure page also yielded no text. These limitations are
recorded in `extraction_report.json`; private-use/replacement glyphs are flagged
per PDF page and retained unchanged for review. No OCR-derived facts were
added. Failed
legacy PRADAN endpoints and older stale routes are documented in the corpus
audit and registry notes and were not substituted with secondary sources.

## Artifact map

- `document_registry.json`: source collection ledger and local file checksums.
- `ontology.json`: entity types, justification, identifying fields, and
  source requirements, including considered-but-excluded types.
- `relations.json`: controlled relation names, endpoint constraints, evidence
  policy, examples, and acceptable evidence.
- `raw/`: downloaded official HTML/PDF documents. The existing Aditya-L1
  booklet is reused at its pre-existing pilot path, referenced by the registry,
  rather than duplicated here.
- `extracted_records.jsonl`: source records retaining document, section,
  subsection, page, and content kind.
- `cleaned_documents.jsonl`: whitespace-normalized records; only exact
  repeated PDF margin lines are removed, and each removal is logged.
- `chunks.jsonl`: semantic, section/page-aware chunks with document ID,
  source URL, authority tier, section, and page metadata.
- `fact_annotations.jsonl`: manually reviewed source claims with exact chunk
  locators. Only direct evidence is used; this file does not use model output
  or benchmark questions.
- `entities.jsonl`, `triples.jsonl`: typed KG-ready facts with document,
  URL, section, chunk ID, page, full source chunk text, and exact excerpt
  provenance. The `source_type` value for the collected facts is
  `official_primary`.
- `extraction_report.json`, `cleaning_report.json`, `chunking_report.json`,
  `corpus_statistics.json`, `relational_coverage.json`,
  `validation_report.json`: reproducibility and coverage outputs.
- `../../reports/phase1_corpus_audit.md`,
  `../../reports/phase1_coverage_report.md`,
  `../../reports/phase1_gap_analysis.md`,
  `../../reports/phase1_validation_report.md`: human-readable Phase 1 reports.

## Reproducing the corpus

Run from the repository root with the project virtual environment. Downloading
uses the registry and only permits the approved ISRO/ISSDC hosts; already
present files are checksummed and reused.

```powershell
.venv\Scripts\python.exe -m src.corpus.download_documents
.venv\Scripts\python.exe -m src.corpus.extract_documents
.venv\Scripts\python.exe -m src.corpus.clean_documents
.venv\Scripts\python.exe -m src.corpus.chunk_documents
.venv\Scripts\python.exe -m src.corpus.build_kg_ready
.venv\Scripts\python.exe -m src.corpus.coverage_analysis
.venv\Scripts\python.exe -m src.corpus.validate_corpus
```

The fact annotations are a reviewed research-data input; regenerating KG-ready
outputs does not automatically extract new facts. To add facts, first inspect
the cited official page/document, add a source-exact locator to
`fact_annotations.jsonl`, then rebuild and validate. A failed validator returns
non-zero and writes details to `validation_report.json`.

## Design and scope boundaries

Chunking targets up to 350 whitespace-delimited words while preserving
document, section, page, and table-row boundaries; it has no overlap and is
not tuned against QA results. The KG records only explicit relations and
source-supported name normalization. In particular, dates alone do not
establish mission-predecessor relations, broad institutional participation
does not establish payload development, and a planned Gaganyaan launcher is
not represented as an actual launch.

Phase 1 ends at a validated provenance-preserving corpus and KG-ready
representation. It does not redesign the benchmark, run model comparisons,
tune retrieval, generate answers, or optimize against evaluation results.
