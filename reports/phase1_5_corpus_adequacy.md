# Phase 1.5 — Corpus adequacy / relational coverage audit

## Decision

**TARGETED COLLECTION REQUIRED**

The current data can support relation-intensive prompts, but breadth is narrow: the core Mission→Payload→DEVELOPED_BY pattern has only 13 payload-level paths across 3 missions; Mission→Payload→OBSERVES has 3 paths in one mission; mission-level PRECEDED_BY has 1 edge; and there are 0 directed three-hop paths. That is a defensible small pilot for selected patterns, but insufficient breadth for a balanced seven-mission relational-augmentation study with temporal and multi-hop coverage.

This conclusion is not a finding that KG augmentation will improve QA. The current corpus supports a bounded pilot comparison on selected relational patterns, but its two-hop and temporal evidence is concentrated and its three-hop coverage is absent. The recommendation is to address the relational evidence/annotation gaps before claiming adequate breadth for the intended seven-mission question family.

First curate explicit facts already present in the Chandrayaan-1 payload PDF and Chandrayaan-2 science-results/overview documents. Then conduct only targeted official-source discovery for Chandrayaan-2/3 payload-specific developer attributions and explicit mission-sequence relations. No new source was verified in this audit, so no unverified document is downloaded or added.

## 1. Verified inventory

- Registered official documents: 33 (www.isro.gov.in: 24, www.issdc.gov.in: 9)
- Authority: {'TIER_1': 33}
- Missions: 7
- Chunks: 1006
- Entities: 94
- Curated triples: 98
- Controlled relations: 12
- Provenance-linked triples: 98/98
- Structural validation: passed (19/19)
- Extraction warnings: 26; pages without extractable text: 9
- Phase 1 statistics cross-checks: {'documents': True, 'chunks': True, 'entities': True, 'triples': True, 'provenance': True, 'relational_coverage_missions': True}

All numbers above were recomputed from the registry, ontology, relation vocabulary, entities, triples, chunks, and reports—not copied from the request. The existing full test suite is separately rerun for this audit.

## 2. Relation frequency and provenance

All relation types present in the current controlled vocabulary are listed. Relations not defined in that vocabulary were not added to the analysis.

| Relation | Triples | Unique subjects | Unique objects | Missions | Distinct source documents | Provenance-supported instances |
|---|---:|---:|---:|---|---:|---:|
| HAS_PAYLOAD | 48 | 6 | 48 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 6 | 48 |
| LAUNCHED_BY | 6 | 6 | 6 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 2 | 6 |
| LAUNCHED_ON | 6 | 6 | 6 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 1 | 6 |
| LAUNCHED_FROM | 6 | 6 | 1 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 1 | 6 |
| HAS_OBJECTIVE | 2 | 2 | 2 | Chandrayaan-3, Gaganyaan | 2 | 2 |
| DEVELOPED_BY | 13 | 13 | 9 | Aditya-L1, AstroSat, Mars Orbiter Mission | 4 | 13 |
| LED_BY | 1 | 1 | 1 | Gaganyaan | 1 | 1 |
| OPERATED_BY | 1 | 1 | 1 | AstroSat | 1 | 1 |
| OBSERVES | 3 | 3 | 2 | Aditya-L1 | 1 | 3 |
| STUDIES | 4 | 4 | 4 | Aditya-L1, AstroSat, Chandrayaan-1, Mars Orbiter Mission | 4 | 4 |
| ORBITS | 7 | 7 | 7 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Gaganyaan, Mars Orbiter Mission | 7 | 7 |
| PRECEDED_BY | 1 | 1 | 1 | Chandrayaan-3 | 1 | 1 |

The document count is the union of provenance document IDs across that relation's triples. All counted edge instances carry provenance. It does not mean the documents contain only that relation.

## 3. Mission × relation matrix

Counts are provenance-supported triple instances, not coverage marks.

| Mission | HAS_PAYLOAD | LAUNCHED_BY | LAUNCHED_ON | LAUNCHED_FROM | HAS_OBJECTIVE | DEVELOPED_BY | LED_BY | OPERATED_BY | OBSERVES | STUDIES | ORBITS | PRECEDED_BY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aditya-L1 | 7 | 1 | 1 | 1 | 0 | 7 | 0 | 0 | 3 | 1 | 1 | 0 |
| AstroSat | 5 | 1 | 1 | 1 | 0 | 5 | 0 | 1 | 0 | 1 | 1 | 0 |
| Chandrayaan-1 | 11 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| Chandrayaan-2 | 13 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Chandrayaan-3 | 7 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Gaganyaan | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| Mars Orbiter Mission | 5 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |

## 4. Provenance-supported two-hop paths

There are **33 distinct directed two-hop path instances** across **8 type/relation patterns**. Each path below requires both edges to have provenance; source evidence is shown for one example per pattern. Full instances and evidence are in `data/corpus/relational_capacity.json`.

| Pattern | Paths | Missions | Union of source documents | Both edges sourced? | Example path and edge evidence |
|---|---:|---|---:|---|---|
| Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> ISROCentre | 5 | Aditya-L1, Mars Orbiter Mission | 3 | yes | Aditya-L1 —HAS_PAYLOAD→ Solar Low Energy X-ray Spectrometer (SoLEXS) / Solar Low Energy X-ray Spectrometer (SoLEXS) —DEVELOPED_BY→ U R Rao Satellite Centre (URSC)<br>ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00007: “Solar Low Energy X-ray Spectrometer (SoLEXS)” ; ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00007: “SoLEXS” |
| Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> Organization | 8 | Aditya-L1, AstroSat | 3 | yes | Aditya-L1 —HAS_PAYLOAD→ Visible Emission Line Coronagraph (VELC) / Visible Emission Line Coronagraph (VELC) —DEVELOPED_BY→ Indian Institute of Astrophysics (IIA)<br>ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00005: “Visible Emission Line Coronagraph(VELC)” ; ISSDC_ADITYA_BOOKLET ISSDC_ADITYA_BOOKLET::c00013: “developed by Indian Institute of Astrophysics” |
| Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> ScientificTarget | 3 | Aditya-L1 | 1 | yes | Aditya-L1 —HAS_PAYLOAD→ Solar Ultraviolet Imaging Telescope (SUIT) / Solar Ultraviolet Imaging Telescope (SUIT) —OBSERVES→ solar photosphere and chromosphere<br>ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00006: “Solar Ultraviolet Imaging Telescope (SUIT)” ; ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00006: “Photosphere and Chromosphere Imaging” |
| Mission -> PRECEDED_BY -> Mission -> HAS_PAYLOAD -> Payload | 13 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —HAS_PAYLOAD→ Terrain Mapping Camera-2 (TMC-2)<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISSDC_CY2_PAYLOADS ISSDC_CY2_PAYLOADS::c00001: “Terrain Mapping Camera-2 (TMC-2)” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_BY -> LaunchVehicle | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_BY→ GSLV-Mk III - M1<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_CY2_LAUNCH_DETAILS ISRO_CY2_LAUNCH_DETAILS::c00002: “GSLV MkIII-M1” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_FROM -> LaunchSite | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_FROM→ Satish Dhawan Space Centre SHAR, Sriharikota<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_LAUNCH_ARCHIVE ISRO_LAUNCH_ARCHIVE::c00033: “Chandrayaan-2” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_ON -> Date | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_ON→ 2019-07-22<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_LAUNCH_ARCHIVE ISRO_LAUNCH_ARCHIVE::c00033: “Jul 22, 2019” |
| Mission -> PRECEDED_BY -> Mission -> ORBITS -> Orbit | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —ORBITS→ circular polar orbit around the Moon<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISSDC_CY2_OVERVIEW ISSDC_CY2_OVERVIEW::c00005: “circular polar orbit around the Moon” |

In addition, **3 shared-payload branching motifs** join a payload's `DEVELOPED_BY` and `OBSERVES` facts. These support the requested form “which organization developed a payload that observes X”; they are not counted as directed Mission-starting paths. All are in Aditya-L1 and have provenance on both facts.

## 5. Three-hop paths

- Provenance-supported directed three-hop paths: **0**
- Three-hop patterns: **0**
- Mission coverage: none.
- Provenance coverage: not applicable; no candidate three-edge chain exists.

The graph does not currently support the example organization → operations-centre/location extension. No relation was inferred to make a longer chain.

## 6. Question-construction capacity (estimate, not a benchmark)

Counts conservatively describe distinct evidence-bounded prompt opportunities or path/fact units. They are not generated benchmark questions and should not be summed across categories as independent samples.

| Category | Conservative prompt capacity (supporting paths/facts) | Missions | Source documents | Independently grounded? |
|---|---|---|---|---|
| A. Direct factual | 6 (supporting paths/facts: 6) | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | ISRO_LAUNCH_ARCHIVE | yes |
| B. Single relationship | 6 (supporting paths/facts: 48) | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_CY1_OVERVIEW, ISRO_CY3_DETAILS, ISRO_MOM_DETAILS, ISSDC_CY2_PAYLOADS | yes |
| C. Relational attribute | 13 (supporting paths/facts: 13) | Aditya-L1, AstroSat, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_MOM_DETAILS, ISSDC_ADITYA_BOOKLET | yes |
| D. Two-hop | 3 (supporting paths/facts: 13) | Aditya-L1, AstroSat, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_MOM_DETAILS, ISSDC_ADITYA_BOOKLET | yes |
| E. Scientific relationship | 2 (supporting paths/facts: 3) | Aditya-L1 | ISRO_ADITYA_OVERVIEW | yes |
| F. Temporal/mission relationship | 1 (supporting paths/facts: 1) | Chandrayaan-3 | ISRO_CY3_DETAILS | yes |
| G. Multi-hop payload/organization/science | 2 (supporting paths/facts: 3) | Aditya-L1 | ISRO_ADITYA_OVERVIEW, ISSDC_ADITYA_BOOKLET | yes |

For the two-hop developer category, 3 grouped mission-level query stems cover 13 distinct payload/developer chains. The scientific relationship capacity is limited to one mission. Corpus-unanswerable prompts are possible in principle, but are not counted: corpus absence does not establish real-world nonexistence, and such questions require explicit corpus-scope wording in a later benchmark protocol.

## 7. KG-necessity research-design classification

These are structural classifications, not performance claims. “Potentially useful” does not mean text retrieval cannot find the facts.

| Relational pattern | Classification | Rationale |
|---|---|---|
| HAS_PAYLOAD / LAUNCHED_BY / LAUNCHED_ON / LAUNCHED_FROM / ORBITS | KG-NOT-NECESSARY | Each is typically stated together in one mission or launch-archive chunk; graph traversal may normalize and join them but is not needed to retrieve a single fact. |
| DEVELOPED_BY (payload → organization) | KG-NOT-NECESSARY | Each supported attribution is directly stated in its source chunk(s); KG can normalize repeated organizations across payloads. |
| Mission → HAS_PAYLOAD → Payload → DEVELOPED_BY → Organization/ISROCentre | KG-MULTI-HOP; potentially KG-beneficial | Requires joining payload membership and developer attribution, often across different source documents/chunks. The graph expresses the question naturally, but text retrieval could retrieve both passages. |
| Mission → HAS_PAYLOAD → Payload → OBSERVES → ScientificTarget | KG-MULTI-HOP; KG-RELATIONALLY-NATURAL | Explicit linked relations form the query structure; currently narrow and within Aditya-L1. |
| Payload → DEVELOPED_BY → Organization and Payload → OBSERVES → Target | KG-POTENTIALLY-USEFUL | A branching join at the same payload supports organization-for-observation questions; facts may be in different booklet/page chunks. |
| PRECEDED_BY | KG-NOT-NECESSARY for the one supported instance | The source explicitly states the follow-on relation in one chunk. Current issue is coverage breadth, not retrieval complexity. |
| Mission → HAS_OBJECTIVE / STUDIES | KG-NOT-NECESSARY in current sources | The objective or science statement is directly present in a chunk; objective triples are under-curated. |
| Any three-edge chain | No current classification instance | There are zero provenance-supported directed three-hop paths in the current triples. |

## 8. Chandrayaan gap investigation

The key distinction is between absent from the curated graph and absent from collected authoritative text.

### Chandrayaan-1

- Developer finding: **B: explicit for some payloads in collected source text but not curated; other wording is ambiguous for DEVELOPED_BY.**
- Objective finding: **B: payload objectives are in the collected ISSDC payload document but not exhaustively represented as triples.**
- Extraction: The payload PDF has text on all 17 pages but extraction flagged private-use glyphs on 10 pages. The cited claims are readable excerpts; unrelated glyphs should not be silently normalized.
- Assessment: The document states five core payload/experiments were indigenously developed and explicitly says C1XS and SARA are developed jointly by ESA and ISRO. The domestic core statement does not assign each payload to a named centre. 'From ... through ESA/NASA' statements do not by themselves prove DEVELOPED_BY.
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “C1XS and SARA are developed by ESA jointly with ISRO”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Rutherford Appleton Laboratory, UK and ISRO Satellite Centre, ISRO”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Near Infra-Red spectrometer (SIR-2) from Max Plank Institute”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00002](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Moon Mineralogy Mapper (M3) from Brown University and Jet Propulsion Laboratory, USA through NASA”

### Chandrayaan-2

- Developer finding: **A: no direct payload-to-developer attribution was found in the collected payload overview, payload document, brochure text, science-results volume, or data handbook during targeted text review.**
- Objective finding: **B: mission objectives and science objectives are directly stated in collected ISSDC science-results text but not curated.**
- Extraction: 150/150 pages of the science-results volume and 53/53 handbook pages yielded text; 7 PDF pages of the payload document have unresolved private-use glyphs.
- Assessment: Collected payload pages name instruments and describe observations/objectives. The overview has empty 'Mission Objectives' and 'Science Objectives' heading chunks, but the 150-page ISSDC science-results volume (and 53-page payload-data handbook) contains explicit mission-level objective paragraphs. No developer attribution was confirmed from direct text in this corpus; author affiliations or centre involvement must not be substituted.
  - [ISSDC_CY2_SCIENCE_RESULTS ISSDC_CY2_SCIENCE_RESULTS::c00035](https://www.issdc.gov.in/docs/ch2/science_results_from_ch-2.pdf): “Mission objectives are as follows”
  - [ISSDC_CY2_SCIENCE_RESULTS ISSDC_CY2_SCIENCE_RESULTS::c00036](https://www.issdc.gov.in/docs/ch2/science_results_from_ch-2.pdf): “The scientific objective of the mission is to expand the lunar scientific knowledge”
  - [ISSDC_CY2_OVERVIEW ISSDC_CY2_OVERVIEW::c00011](https://www.issdc.gov.in/chandrayaan2.html): “science payloads aim to perform detailed study of lunar topography”

### Chandrayaan-3

- Developer finding: **A: no explicit payload developer attribution found in the collected mission-detail page, 12-page brochure, and overview text.**
- Objective finding: **B: mission objective and payload objectives are directly stated in the collected details page but not broadly curated.**
- Extraction: The collected overview, details, and brochure were text-extracted without recorded empty pages or warnings.
- Assessment: Mission details explicitly enumerate payloads and objectives and call the lander/propulsion module/rover indigenous, but do not attribute individual payload development to named centres. General ISRO ownership or project participation is not a payload-specific developer edge.
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002](https://www.isro.gov.in/Chandrayaan3_Details.html): “demonstrate end-to-end capability in safe landing and roving on the lunar surface”
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00023](https://www.isro.gov.in/Chandrayaan3_Details.html): “Objectives: To carry out the measurements of thermal properties”
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00024](https://www.isro.gov.in/Chandrayaan3_Details.html): “Objectives: To measure seismicity around the landing site”


Summary: CY1 has some explicit payload developer evidence and multiple payload objectives already collected but uncurated; more general domestic/international affiliation phrases are ambiguous for `DEVELOPED_BY`. CY2 objectives are present in collected ISSDC sources but uncurated; specific payload developer attribution was not found in reviewed extracted text. CY3 objectives are present in the collected details page but uncurated; payload-specific developer attribution was not found. These facts do **not** establish that no official developer documentation exists elsewhere.

## 9. Additional official sources and access limits

No additional source was verified sufficiently to enter the registry or be downloaded during this audit. Attempted official ISRO/ISSDC site-search routes returned 404; other search endpoints did not yield inspectable primary documents. No search snippet is treated as evidence.

The targeted discovery list is therefore:

| Priority | Required source/document | Organization / mission | Specific relationship gap | Candidate URL / title status | Why current corpus is insufficient |
|---|---|---|---|---|---|
| 1 | Payload-specific development/realization source naming responsible centre/institution for Chandrayaan-2 payloads | ISRO/ISSDC; Chandrayaan-2 | `Payload DEVELOPED_BY Organization/ISROCentre` | Official document/title/URL not yet verified; do not download until opened and exact language checked. | Collected payload documentation and result volumes support payload identity/objectives, but targeted text review found no safe named developer attribution. |
| 1 | Payload-specific development/realization source for Chandrayaan-3 payloads | ISRO; Chandrayaan-3 | `Payload DEVELOPED_BY Organization/ISROCentre` | Official document/title/URL not yet verified; do not download until opened and exact language checked. | Collected details and brochure list payloads/objectives but do not attribute payload-specific development. |
| 2 | Primary mission-history source with explicit Chandrayaan-1 → Chandrayaan-2 programme sequence language | ISRO/ISSDC; Chandrayaan-1/2 | `PRECEDED_BY` | No additional source verified. | Existing evidence discusses inherited technologies and follow-on payloads, but only one mission-level predecessor edge is curated; chronology alone is insufficient. |

### In-corpus annotation before any download

The targeted evidence review also found high-value directly supported claims already inside collected documents:

- Chandrayaan-1: ISSDC payload PDF, page 1 (`ISSDC_CY1_PAYLOADS::c00001`) explicitly says C1XS and SARA were developed jointly by ESA and ISRO; the same PDF contains objectives for payloads. Curate only claims with unambiguous entity identity and wording. “From institution through agency” is not automatically a developer claim.
- Chandrayaan-2: ISSDC science-results volume, page 19 (`ISSDC_CY2_SCIENCE_RESULTS::c00035` and `::c00036`) states mission and science objectives. No new document is required to add those direct objective facts.
- Chandrayaan-3: ISRO details page (`ISRO_CY3_DETAILS::c00002` and payload-objective table rows) states mission/payload objectives. No new document is required for those facts.

Thus the first action is a bounded human annotation review, not indiscriminate collection. The Priority 1 document search is warranted only for the unresolved CY2/CY3 developer links and should stop if no directly explicit authoritative records are found.

## 10. Final rationale and boundaries

The existing corpus is authoritative and structurally provenance-complete for its 98 currently curated triples. It can support direct facts, payload membership across six flown missions, developer attributes in three missions, one temporal relation, and several two-hop joins. Yet raw document/chunk count overstates usable graph connectivity: the 33 directed two-hop paths include many paths that merely extend the single `Chandrayaan-3 PRECEDED_BY Chandrayaan-2` edge into arbitrary launch/payload facts. The intended payload-developer joins are 13 paths; observation joins are 3 paths; there are zero three-hop paths. Temporal evidence is a single mission pair.

Accordingly, this audit does **not** recommend a broad corpus expansion. It recommends **TARGETED COLLECTION REQUIRED** only for explicit missing Chandrayaan-2/3 payload developers and, secondarily, direct programme-sequence evidence. Re-annotate direct facts already found in the corpus first. If official searches reveal no suitable sources, preserve the coverage limitation and either narrow later evaluation claims/questions to supported patterns or report the limitation; do not invent relations or treat incomplete coverage as a real-world negative.

No benchmark, benchmark answer, result file, prior experimental report, or paper was modified or run as part of this audit.
