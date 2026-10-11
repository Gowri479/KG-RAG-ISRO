# KG-RAG diagnostic audit

## Scope

This audit inspects the implementation used by the frozen relational
evaluation. It does not modify the benchmark, corpus, prior results, or paper.

## Existing pipeline

1. `src/retriever/hybrid.py` extracts entities with spaCy `en_core_web_lg`
   plus a regex fallback. It then filters entities using query keywords.
2. `src/retriever/query_classifier.py` classifies the query with regular
   expressions. Temporal matching runs before multi-hop matching, so a query
   containing a temporal marker can be assigned a one-hop temporal type even
   when it is compositional.
3. `src/retriever/kg_retriever.py` loads
   `data/kg/knowledge_graph.pkl` first. This is a large pilot graph
   (31,358 nodes and 95,671 edges) containing dependency-style relations and
   many empty-source edges. The canonical Phase 1 corpus graph is represented
   separately by `data/corpus/triples.jsonl` (134 curated triples).
4. The frozen Phase 4 evaluator calls `_get_one_hop` directly for each filtered
   entity. It does not call the controlled two-hop retriever.
5. The existing KG context is serialized as unlabelled natural-language
   sentences. Relation names are lower-cased and underscores become spaces.
6. Dense passages come from a separate pilot FAISS index and pilot chunk list.
   The frozen corpus chunk IDs are not preserved by this retrieval path.
7. `src/retriever/hybrid.py` concatenates KG text and dense text, then truncates
   by whitespace. It does not preserve a stable evidence-item order, path ID,
   or source-linked canonical triple ID in the final context.
8. `src/generator/ollama_api.py` passes the merged context to the existing
   prompt and Mistral configuration.

## Main weaknesses

### Runtime graph mismatch

The frozen evaluator reads the pilot `knowledge_graph.pkl`, not the
provenance-linked Phase 1 graph in `data/corpus/triples.jsonl`. Exact matching
of the 26 frozen KG-required questions against canonical subject/object/relation
edges found no matching runtime edges. This makes the old path-retrieval trace
not an interpretable test of the frozen corpus KG.

### Entity and relation resolution

Entity extraction returns surface strings and does not consistently apply the
canonical aliases in `src/kg_builder/relations.py` before graph lookup.
Organization names and payload names therefore frequently fail exact node
lookup. Relation selection is implicit in graph neighbourhood expansion rather
than a controlled relation query.

### Hop depth

The frozen Phase 4 `_kg_trace` uses `_get_one_hop` unconditionally and records
`hop_depth=1` whenever any triple is returned. The controlled two-hop module
exists but is not used by that evaluator.

### Provenance loss

The pilot graph frequently stores only `source`, often empty, and the old
trace serializes triples without canonical triple IDs or source chunk IDs.
Consequently, `retrieved_kg_paths` was hard-coded to an empty list in the
frozen evaluator. A correct KG edge cannot be connected reliably to a frozen
corpus source from those traces.

### Evidence fusion

KG and dense evidence are concatenated as free text. There is no explicit
entity/relation/path structure, no source block for KG facts, and no
source-aware deduplication. Whitespace truncation can remove evidence from the
end of the merged context.

## Diagnostic implication

The negative Phase 4 result cannot distinguish technical KG retrieval failure,
provenance/fusion failure, generator-use failure, and a genuine lack of KG
benefit. A separate diagnostic pipeline must use the existing canonical
provenance-linked triples and chunks, preserve path IDs, and expose the exact
evidence sent to the generator. The controlled diagnostic implementation is
isolated from the original KG-RAG path and uses the same frozen questions and
Mistral generation configuration.
