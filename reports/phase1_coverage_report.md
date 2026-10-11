# Phase 1 corpus coverage

## Corpus inventory

- Registered documents: 33
- Semantic chunks: 1006
- Estimated words: 65,601
- Missions: 7
- Typed entities: 123
- Payload entities: 48
- Organization/ISRO-centre entities: 12
- Launch-vehicle entities: 6
- Orbit entities: 7
- Curated triples: 134
- Controlled relation types represented: 12 of 12
- Provenance-linked triples: 134 of 134

All registered documents are Tier 1 official ISRO or ISSDC sources. Word counts are whitespace-token estimates, not model token counts. Chunking and annotation were completed without use of benchmark questions, model answers, or downstream QA scores.

| Authority tier | Documents |
|---|---:|
| TIER_1 | 33 |

## Mission coverage

| Mission | Documents | Chunks | Payload entities | Payloads with explicit developer | Orbit relations | Triples |
|---|---:|---:|---:|---:|---:|---:|
| Aditya-L1 | 3 | 40 | 7 | 7 | 1 | 22 |
| AstroSat | 2 | 9 | 5 | 5 | 1 | 16 |
| Chandrayaan-1 | 5 | 95 | 11 | 2 | 1 | 33 |
| Chandrayaan-2 | 7 | 580 | 13 | 0 | 1 | 28 |
| Chandrayaan-3 | 3 | 79 | 7 | 0 | 1 | 21 |
| Gaganyaan | 2 | 6 | 0 | 0 | 1 | 3 |
| Mars Orbiter Mission | 2 | 43 | 5 | 1 | 1 | 11 |

## Relations represented

| Relation | Triples |
|---|---:|
| DEVELOPED_BY | 17 |
| HAS_OBJECTIVE | 26 |
| HAS_PAYLOAD | 48 |
| LAUNCHED_BY | 6 |
| LAUNCHED_FROM | 6 |
| LAUNCHED_ON | 6 |
| LED_BY | 1 |
| OBSERVES | 9 |
| OPERATED_BY | 1 |
| ORBITS | 7 |
| PRECEDED_BY | 1 |
| STUDIES | 6 |

## Relational coverage matrix

| Mission | Payload | Developer organization | Launch | Orbit | Temporal | Science/objective |
|---|---|---|---|---|---|---|
| Aditya-L1 | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| AstroSat | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Chandrayaan-1 | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Chandrayaan-2 | ✓ | — | ✓ | ✓ | — | ✓ |
| Chandrayaan-3 | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Gaganyaan | — | — | N/A (unflown) | ✓ | — | ✓ |
| Mars Orbiter Mission | ✓ | ✓ | ✓ | ✓ | — | ✓ |

`✓` means at least one directly supported fact is curated, not that the category is exhaustively covered. A dash means no such fact is currently represented and is not a negative factual claim. Orbit status distinguishes occupied from planned/targeted in entity properties and annotation notes. Gaganyaan launch is not applicable to the unflown crewed mission; its planned launcher is not mislabeled as an actual launch.
