# Canonical 180-question KG-relevance annotation guidelines

## Scope and frozen inputs

The annotation covers only the 180 question IDs listed in
`data/benchmark/test_ids.json`, joined by ID to the 200-item
`data/benchmark/isro_qa.json` bank and the Phase 1.6 frozen graph in
`data/corpus/triples.jsonl`. The separate 36-question Aditya-L1 stress test is
not included. No benchmark item, answer, ID, or split is changed.

The benchmark records do not provide a mission field. `mission` is conservatively
derived from named mission aliases in the question text. It is one canonical
mission name, `multiple: ...` for multiple named missions, or
`not mission-specific`. Mission names are not inferred from a reference answer.

Labels are assigned from question wording, requested answer structure, and
controlled KG relations/evidence. Stored model outputs, scores, and system
identity are not inputs to annotation.

## Primary question type (exactly one)

- `DIRECT_FACT`: a single explicit fact, name, date, count, or definition.
- `ATTRIBUTE`: one property of a named entity, such as its orbit, objective,
  mass, or mission life.
- `SINGLE_RELATION`: one entity-to-entity association, including a relationship
  not currently instantiated in the corpus.
- `TWO_HOP_RELATION`: a linked chain in which the answer depends on a mission,
  its payload, and a payload-associated developer, objective, target, or result.
- `MULTI_RELATION`: multiple facts/evidence items, independent attributes, or
  comparisons that do not form one linked two-edge path. Comparing dates or
  asking for several properties is not automatically KG multi-hop.
- `OTHER`: use only when the question does not fit the categories above.

If a question asks several things, classify it by the most structurally
demanding required answer. A direct definition plus an independent observation
is `MULTI_RELATION`; a mission→payload→developer chain is
`TWO_HOP_RELATION`.

## KG relevance (exactly one)

- `KG_NOT_RELEVANT`: the answer is a direct fact, unmodeled relation, or simple
  date/arithmetic comparison for which this graph structure is not naturally
  needed. A date triple existing in the KG does not alone make a date question
  KG-relevant.
- `KG_POTENTIALLY_USEFUL`: graph structure may organize related facts, but one
  passage can reasonably answer the question, or the exact requested edge is
  missing/incomplete in this corpus.
- `KG_RELATIONALLY_NATURAL`: wording asks for a relationship instantiated in
  the controlled KG, such as a mission launcher or payload-target relation.
- `KG_MULTI_HOP`: the answer structure corresponds to joining at least two
  explicit, provenance-supported KG edges. The label denotes graph structure,
  not proof that text retrieval cannot retrieve both facts.

The intended multi-hop patterns include mission→payload→developer,
mission→payload→objective, and mission→payload→observation. An objective edge
is not treated as evidence that a historical result occurred; an observation
target edge is not treated as evidence that a discovery was made.

## KG-required decision rule

- `YES` only when the answer fundamentally requires joining explicit relational
  facts and no single relevant passage is a reasonable answer source.
- `NO` when ordinary text retrieval can reasonably answer from one relevant
  passage, when the query is a single fact/relation, or when the question does
  not depend on a KG join.
- `UNCERTAIN` when the available benchmark and corpus artifacts cannot
  establish whether a consolidated passage suffices.

Do not infer `YES` from a corresponding triple, a multi-hop question type, or
KG-RAG's performance. Do not infer `NO` from Vanilla RAG performance. The saved
canonical result rows preserve generated answers but not retrieved passages or
contexts, so ambiguous single-passage cases remain `UNCERTAIN`.

## Relation names and evidence

`kg_relation_types` uses only names in `data/corpus/relations.json`. When a
question requests a relation whose specific entity edge is missing, the
controlled relation name may still be recorded, but the missing edge is not
invented. `supporting_evidence` cites only existing graph triples and includes
triple ID, endpoints, relation, source document, and source chunk.

Evidence is a relevance anchor, not a guarantee that the whole question or
reference answer is supported. In particular:

- A matching relation type with a different entity pair is not evidence for
  the requested fact.
- A partial path may be cited only with a reason explaining the missing edge.
- `PRECEDED_BY` is not inferred from `LAUNCHED_ON` date ordering.
- No source or triple is added to the frozen corpus by this annotation.

## Ambiguous cases and audit notes

Questions asking for a developer that is absent from the graph, multiple
historical findings, or a relationship only partly covered by an existing path
are not upgraded to KG-multi-hop without the corresponding explicit edges.
Such records use a potentially-useful label with a gap explanation, or
`UNCERTAIN` for `kg_required` when the one-passage distinction cannot be made.

The annotation is single-researcher and has no inter-annotator statistic.
Performance results are analyzed only after labels are generated and validated.
ROUGE-L, reference-token coverage, exact match, and IDK are lexical/abstention
metrics; they do not establish factual correctness.
