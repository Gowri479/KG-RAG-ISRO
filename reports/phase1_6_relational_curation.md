# Phase 1.6 — Existing-evidence relational curation and adequacy

## Decision

**TARGETED COLLECTION REQUIRED**

The current data can support relation-intensive prompts, but breadth is narrow: the core Mission→Payload→DEVELOPED_BY pattern has only 17 payload-level paths across 4 missions; Mission→Payload→OBSERVES has 9 paths across 4 missions; Mission→Payload→HAS_OBJECTIVE has 21 paths across 3 missions; mission-level PRECEDED_BY has 1 edge; and 8 directed three-hop paths are all rooted in 8 continuations of that same temporal edge. The curated payload/objective and observation paths now provide meaningful two-hop coverage, but one explicit temporal edge is not enough to support a temporal question family. Since no additional authoritative sequence source was verified, keep temporal QA out of scope or perform narrowly targeted collection before including it.

This is a corpus-design decision, not a finding that KG augmentation improves QA. Phase 1.6 curated explicit facts already present in collected Chandrayaan sources. It did not add documents, create benchmark questions, run experiments, modify results, or edit the paper.

Explicit Chandrayaan-1/2/3 objective evidence already present in collected sources has been curated; no additional corpus documents were downloaded. Further collection is limited to an official source explicitly documenting multiple mission-sequence relations, if temporal questions remain in scope. No new source was verified in this audit, so no unverified document is downloaded or added.

## 1. Verified inventory

- Registered official documents: 33 (www.isro.gov.in: 24, www.issdc.gov.in: 9)
- Authority: {'TIER_1': 33}
- Missions: 7
- Chunks: 1006
- Entities: 123
- Curated triples: 134
- Controlled relations: 12
- Provenance-linked triples: 134/134
- Structural validation: passed (19/19)
- Extraction warnings: 26; pages without extractable text: 9
- Phase 1 statistics cross-checks: {'documents': True, 'chunks': True, 'entities': True, 'triples': True, 'provenance': True, 'relational_coverage_missions': True}
- Phase 1.5 → Phase 1.6 changes: {'registered_document_count': 33, 'chunk_count': 1006, 'entity_count': 94, 'triple_count': 98, 'additional_documents_in_phase1_6': 0, 'additional_chunks_in_phase1_6': 0, 'additional_entities_in_phase1_6': 29, 'additional_triples_in_phase1_6': 36}
- New curated triples by relation: {'HAS_OBJECTIVE': 24, 'DEVELOPED_BY': 4, 'OBSERVES': 6, 'STUDIES': 2}

All numbers above were recomputed from the registry, ontology, relation vocabulary, entities, triples, chunks, and reports—not copied from the request. The existing full test suite is separately rerun for this audit.

## 2. Relation frequency and provenance

All relation types present in the current controlled vocabulary are listed. Relations not defined in that vocabulary were not added to the analysis.

| Relation | Triples | Unique subjects | Unique objects | Missions | Distinct source documents | Provenance-supported instances |
|---|---:|---:|---:|---|---:|---:|
| HAS_PAYLOAD | 48 | 6 | 48 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 6 | 48 |
| LAUNCHED_BY | 6 | 6 | 6 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 2 | 6 |
| LAUNCHED_ON | 6 | 6 | 6 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 1 | 6 |
| LAUNCHED_FROM | 6 | 6 | 1 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | 1 | 6 |
| HAS_OBJECTIVE | 26 | 24 | 26 | Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Gaganyaan | 5 | 26 |
| DEVELOPED_BY | 17 | 15 | 10 | Aditya-L1, AstroSat, Chandrayaan-1, Mars Orbiter Mission | 5 | 17 |
| LED_BY | 1 | 1 | 1 | Gaganyaan | 1 | 1 |
| OPERATED_BY | 1 | 1 | 1 | AstroSat | 1 | 1 |
| OBSERVES | 9 | 9 | 7 | Aditya-L1, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3 | 4 | 9 |
| STUDIES | 6 | 6 | 4 | Aditya-L1, AstroSat, Chandrayaan-1, Mars Orbiter Mission | 5 | 6 |
| ORBITS | 7 | 7 | 7 | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Gaganyaan, Mars Orbiter Mission | 7 | 7 |
| PRECEDED_BY | 1 | 1 | 1 | Chandrayaan-3 | 1 | 1 |

The document count is the union of provenance document IDs across that relation's triples. All counted edge instances carry provenance. It does not mean the documents contain only that relation.

## 3. Mission × relation matrix

Counts are provenance-supported triple instances, not coverage marks.

| Mission | HAS_PAYLOAD | LAUNCHED_BY | LAUNCHED_ON | LAUNCHED_FROM | HAS_OBJECTIVE | DEVELOPED_BY | LED_BY | OPERATED_BY | OBSERVES | STUDIES | ORBITS | PRECEDED_BY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aditya-L1 | 7 | 1 | 1 | 1 | 0 | 7 | 0 | 0 | 3 | 1 | 1 | 0 |
| AstroSat | 5 | 1 | 1 | 1 | 0 | 5 | 0 | 1 | 0 | 1 | 1 | 0 |
| Chandrayaan-1 | 11 | 1 | 1 | 1 | 9 | 4 | 0 | 0 | 2 | 3 | 1 | 0 |
| Chandrayaan-2 | 13 | 1 | 1 | 1 | 9 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| Chandrayaan-3 | 7 | 1 | 1 | 1 | 7 | 0 | 0 | 0 | 2 | 0 | 1 | 1 |
| Gaganyaan | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| Mars Orbiter Mission | 5 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |

## 4. Provenance-supported two-hop paths

There are **69 distinct directed two-hop path instances** across **13 type/relation patterns**. Each path below requires both edges to have provenance; source evidence is shown for one example per pattern. Full instances and evidence are in `data/corpus/relational_capacity.json`.

| Pattern | Paths | Missions | Union of source documents | Both edges sourced? | Example path and edge evidence |
|---|---:|---|---:|---|---|
| Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> ISROCentre | 5 | Aditya-L1, Mars Orbiter Mission | 3 | yes | Aditya-L1 —HAS_PAYLOAD→ Solar Low Energy X-ray Spectrometer (SoLEXS) / Solar Low Energy X-ray Spectrometer (SoLEXS) —DEVELOPED_BY→ U R Rao Satellite Centre (URSC)<br>ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00007: “Solar Low Energy X-ray Spectrometer (SoLEXS)” ; ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00007: “SoLEXS” |
| Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> Organization | 12 | Aditya-L1, AstroSat, Chandrayaan-1 | 5 | yes | Chandrayaan-1 —HAS_PAYLOAD→ Chandrayaan-I X-ray Spectrometer (CIXS) / Chandrayaan-I X-ray Spectrometer (CIXS) —DEVELOPED_BY→ ESA<br>ISRO_CY1_OVERVIEW ISRO_CY1_OVERVIEW::c00008: “CIXS” ; ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001: “C1XS and SARA are developed by ESA jointly with ISRO” |
| Mission -> HAS_PAYLOAD -> Payload -> HAS_OBJECTIVE -> ScientificObjective | 21 | Chandrayaan-1, Chandrayaan-2, Chandrayaan-3 | 4 | yes | Chandrayaan-1 —HAS_PAYLOAD→ Terrain Mapping Camera (TMC) / Terrain Mapping Camera (TMC) —HAS_OBJECTIVE→ Map the near and far sides of the Moon and prepare a three-dimensional lunar topographic atlas.<br>ISRO_CY1_OVERVIEW ISRO_CY1_OVERVIEW::c00007: “Terrain Mapping Camera (TMC)” ; ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00006: “map topography of both near and far side of the Moon” |
| Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> Phenomenon | 3 | Chandrayaan-1, Chandrayaan-2 | 3 | yes | Chandrayaan-1 —HAS_PAYLOAD→ Sub keV Atom Reflecting Analyzer (SARA) / Sub keV Atom Reflecting Analyzer (SARA) —OBSERVES→ solar-wind interaction with the lunar surface<br>ISRO_CY1_OVERVIEW ISRO_CY1_OVERVIEW::c00008: “SARA” ; ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00028: “Imaging the solar wind-surface interaction” |
| Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> PlanetaryBody | 3 | Chandrayaan-2, Chandrayaan-3 | 2 | yes | Chandrayaan-2 —HAS_PAYLOAD→ Alpha Particle X-ray Spectrometer (APXS) / Alpha Particle X-ray Spectrometer (APXS) —OBSERVES→ Moon<br>ISSDC_CY2_PAYLOADS ISSDC_CY2_PAYLOADS::c00025: “Alpha Particle X -ray Spectrometer (APXS)” ; ISSDC_CY2_PAYLOADS ISSDC_CY2_PAYLOADS::c00025: “elemental composition of the lunar surface” |
| Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> ScientificTarget | 3 | Aditya-L1 | 1 | yes | Aditya-L1 —HAS_PAYLOAD→ Solar Ultraviolet Imaging Telescope (SUIT) / Solar Ultraviolet Imaging Telescope (SUIT) —OBSERVES→ solar photosphere and chromosphere<br>ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00006: “Solar Ultraviolet Imaging Telescope (SUIT)” ; ISRO_ADITYA_OVERVIEW ISRO_ADITYA_OVERVIEW::c00006: “Photosphere and Chromosphere Imaging” |
| Mission -> HAS_PAYLOAD -> Payload -> STUDIES -> PlanetaryBody | 2 | Chandrayaan-1 | 2 | yes | Chandrayaan-1 —HAS_PAYLOAD→ Terrain Mapping Camera (TMC) / Terrain Mapping Camera (TMC) —STUDIES→ Moon<br>ISRO_CY1_OVERVIEW ISRO_CY1_OVERVIEW::c00007: “Terrain Mapping Camera (TMC)” ; ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00006: “map topography of both near and far side of the Moon” |
| Mission -> PRECEDED_BY -> Mission -> HAS_OBJECTIVE -> ScientificObjective | 3 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —HAS_OBJECTIVE→ Develop and demonstrate newer technologies useful for future planetary missions.<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISSDC_CY2_SCIENCE_RESULTS ISSDC_CY2_SCIENCE_RESULTS::c00035: “newer technologies that will be useful” |
| Mission -> PRECEDED_BY -> Mission -> HAS_PAYLOAD -> Payload | 13 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —HAS_PAYLOAD→ Terrain Mapping Camera-2 (TMC-2)<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISSDC_CY2_PAYLOADS ISSDC_CY2_PAYLOADS::c00001: “Terrain Mapping Camera-2 (TMC-2)” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_BY -> LaunchVehicle | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_BY→ GSLV-Mk III - M1<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_CY2_LAUNCH_DETAILS ISRO_CY2_LAUNCH_DETAILS::c00002: “GSLV MkIII-M1” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_FROM -> LaunchSite | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_FROM→ Satish Dhawan Space Centre SHAR, Sriharikota<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_LAUNCH_ARCHIVE ISRO_LAUNCH_ARCHIVE::c00033: “Chandrayaan-2” |
| Mission -> PRECEDED_BY -> Mission -> LAUNCHED_ON -> Date | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —LAUNCHED_ON→ 2019-07-22<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISRO_LAUNCH_ARCHIVE ISRO_LAUNCH_ARCHIVE::c00033: “Jul 22, 2019” |
| Mission -> PRECEDED_BY -> Mission -> ORBITS -> Orbit | 1 | Chandrayaan-3 | 2 | yes | Chandrayaan-3 —PRECEDED_BY→ Chandrayaan-2 / Chandrayaan-2 —ORBITS→ circular polar orbit around the Moon<br>ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002: “follow-on mission to Chandrayaan-2” ; ISSDC_CY2_OVERVIEW ISSDC_CY2_OVERVIEW::c00005: “circular polar orbit around the Moon” |

In addition, **5 shared-payload branching motifs** join a payload's `DEVELOPED_BY` and `OBSERVES` facts. These support the form “which organization developed a payload that observes X”; they are not counted as directed Mission-starting paths. Missions represented: Aditya-L1, Chandrayaan-1. Both facts in every motif have provenance.

## 5. Three-hop paths

- Provenance-supported directed three-hop paths: **8**
- Three-hop patterns: **3**
- Mission coverage: Chandrayaan-3.
- Provenance coverage: Every enumerated path requires provenance on every edge; 8/8 paths are provenance-supported.

| Pattern | Paths | Missions | Source documents | All edges sourced? |
|---|---:|---|---:|---|
| Mission -> PRECEDED_BY -> Mission -> HAS_PAYLOAD -> Payload -> HAS_OBJECTIVE -> ScientificObjective | 6 | Chandrayaan-3 | 2 | yes |
| Mission -> PRECEDED_BY -> Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> Phenomenon | 1 | Chandrayaan-3 | 2 | yes |
| Mission -> PRECEDED_BY -> Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> PlanetaryBody | 1 | Chandrayaan-3 | 2 | yes |

The graph does not currently support the example organization → operations-centre/location extension. No relation was inferred to make a longer chain.

## 6. Question-construction capacity (estimate, not a benchmark)

Counts conservatively describe distinct evidence-bounded prompt opportunities or path/fact units. They are not generated benchmark questions and should not be summed across categories as independent samples.

| Category | Conservative prompt capacity (supporting paths/facts) | Missions | Source documents | Independently grounded? |
|---|---|---|---|---|
| A. Direct factual | 6 (supporting paths/facts: 6) | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | ISRO_LAUNCH_ARCHIVE | yes |
| B. Single relationship | 6 (supporting paths/facts: 48) | Aditya-L1, AstroSat, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_CY1_OVERVIEW, ISRO_CY3_DETAILS, ISRO_MOM_DETAILS, ISSDC_CY2_PAYLOADS | yes |
| C. Relational attribute | 17 (supporting paths/facts: 17) | Aditya-L1, AstroSat, Chandrayaan-1, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_MOM_DETAILS, ISSDC_ADITYA_BOOKLET, ISSDC_CY1_PAYLOADS | yes |
| D. Two-hop | 4 (supporting paths/facts: 17) | Aditya-L1, AstroSat, Chandrayaan-1, Mars Orbiter Mission | ISRO_ADITYA_OVERVIEW, ISRO_ASTROSAT_DETAILS, ISRO_CY1_OVERVIEW, ISRO_MOM_DETAILS, ISSDC_ADITYA_BOOKLET, ISSDC_CY1_PAYLOADS | yes |
| E. Scientific relationship | 8 (supporting paths/facts: 9) | Aditya-L1, Chandrayaan-1, Chandrayaan-2, Chandrayaan-3 | ISRO_ADITYA_OVERVIEW, ISRO_CY3_DETAILS, ISSDC_CY1_PAYLOADS, ISSDC_CY2_PAYLOADS | yes |
| F. Temporal/mission relationship | 1 (supporting paths/facts: 1) | Chandrayaan-3 | ISRO_CY3_DETAILS | yes |
| G. Multi-hop payload/organization/science | 3 (supporting paths/facts: 5) | Aditya-L1, Chandrayaan-1 | ISRO_ADITYA_OVERVIEW, ISSDC_ADITYA_BOOKLET, ISSDC_CY1_PAYLOADS | yes |
| H. Mission → payload → objective | 3 (supporting paths/facts: 21) | Chandrayaan-1, Chandrayaan-2, Chandrayaan-3 | ISRO_CY1_OVERVIEW, ISRO_CY3_DETAILS, ISSDC_CY1_PAYLOADS, ISSDC_CY2_PAYLOADS | yes |

For the two-hop developer category, 4 grouped mission-level query stems cover 17 distinct payload/developer chains. Objective and observation chains are separately estimated above. Corpus-unanswerable prompts are possible in principle, but are not counted: corpus absence does not establish real-world nonexistence, and such questions require explicit corpus-scope wording in a later benchmark protocol.

## 7. KG-necessity research-design classification

These are structural classifications, not performance claims. “Potentially useful” does not mean text retrieval cannot find the facts.

| Relational pattern | Classification | Rationale |
|---|---|---|
| HAS_PAYLOAD / LAUNCHED_BY / LAUNCHED_ON / LAUNCHED_FROM / ORBITS | KG-NOT-NECESSARY | Each is typically stated together in one mission or launch-archive chunk; graph traversal may normalize and join them but is not needed to retrieve a single fact. |
| DEVELOPED_BY (payload → organization) | KG-NOT-NECESSARY | Each supported attribution is directly stated in its source chunk(s); KG can normalize repeated organizations across payloads. |
| Mission → HAS_PAYLOAD → Payload → DEVELOPED_BY → Organization/ISROCentre | KG-MULTI-HOP; potentially KG-beneficial | Requires joining payload membership and developer attribution, often across different source documents/chunks. The graph expresses the question naturally, but text retrieval could retrieve both passages. |
| Mission → HAS_PAYLOAD → Payload → OBSERVES → target/body/phenomenon | KG-MULTI-HOP; KG-RELATIONALLY-NATURAL | Explicit linked relations form the query structure across the missions and targets listed above; this is a structural classification, not a claim text retrieval cannot combine the evidence. |
| Payload → DEVELOPED_BY → Organization and Payload → OBSERVES → Target | KG-POTENTIALLY-USEFUL | A branching join at the same payload supports organization-for-observation questions; facts may be in different booklet/page chunks. |
| PRECEDED_BY | KG-NOT-NECESSARY for the one supported instance | The source explicitly states the follow-on relation in one chunk. Current issue is coverage breadth, not retrieval complexity. |
| Mission → HAS_PAYLOAD → Payload → HAS_OBJECTIVE → ScientificObjective | KG-MULTI-HOP; potentially KG-beneficial | Joins mission membership with a payload objective; source material can span a mission payload list and separate objective text. Both edges are directly sourced. |
| Mission → HAS_OBJECTIVE / STUDIES | KG-NOT-NECESSARY for a directly stated single fact | The objective or science statement is directly present in a chunk; graph normalization may still aid cross-document joins. |
| PRECEDED_BY → HAS_PAYLOAD → HAS_OBJECTIVE/OBSERVES | KG-MULTI-HOP, narrowly supported | Eight sourced paths exist, but every one starts from the same Chandrayaan-3 → Chandrayaan-2 predecessor edge; they do not provide broad temporal coverage. |

## 8. Chandrayaan gap investigation

The key distinction is between absent from the curated graph and absent from collected authoritative text.

### Chandrayaan-1

- Developer finding: **B: explicit C1XS/SARA statements in collected source text are now curated; other wording remains ambiguous for DEVELOPED_BY.**
- Objective finding: **B: payload objectives already present in the collected ISSDC payload document are now curated with chunk-level provenance.**
- Extraction: The payload PDF has text on all 17 pages but extraction flagged private-use glyphs on 10 pages. The cited claims are readable excerpts; unrelated glyphs should not be silently normalized.
- Assessment: The document states five core payload/experiments were indigenously developed and explicitly says C1XS and SARA are developed jointly by ESA and ISRO. The domestic core statement does not assign each payload to a named centre. 'From ... through ESA/NASA' statements do not by themselves prove DEVELOPED_BY.
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “C1XS and SARA are developed by ESA jointly with ISRO”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Rutherford Appleton Laboratory, UK and ISRO Satellite Centre, ISRO”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00001](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Near Infra-Red spectrometer (SIR-2) from Max Plank Institute”
  - [ISSDC_CY1_PAYLOADS ISSDC_CY1_PAYLOADS::c00002](https://www.issdc.gov.in/docs/ch1/chandrayaan1_payload.pdf): “Moon Mineralogy Mapper (M3) from Brown University and Jet Propulsion Laboratory, USA through NASA”

### Chandrayaan-2

- Developer finding: **A: no direct payload-to-developer attribution was found in the collected payload overview, payload document, brochure text, science-results volume, or data handbook during targeted text review.**
- Objective finding: **B: mission/science objectives and selected payload objectives already present in collected ISSDC text are now curated with chunk-level provenance.**
- Extraction: 150/150 pages of the science-results volume and 53/53 handbook pages yielded text; 7 PDF pages of the payload document have unresolved private-use glyphs.
- Assessment: Collected payload pages name instruments and describe observations/objectives. The overview has empty 'Mission Objectives' and 'Science Objectives' heading chunks, but the 150-page ISSDC science-results volume (and 53-page payload-data handbook) contains explicit mission-level objective paragraphs. No developer attribution was confirmed from direct text in this corpus; author affiliations or centre involvement must not be substituted.
  - [ISSDC_CY2_SCIENCE_RESULTS ISSDC_CY2_SCIENCE_RESULTS::c00035](https://www.issdc.gov.in/docs/ch2/science_results_from_ch-2.pdf): “Mission objectives are as follows”
  - [ISSDC_CY2_SCIENCE_RESULTS ISSDC_CY2_SCIENCE_RESULTS::c00036](https://www.issdc.gov.in/docs/ch2/science_results_from_ch-2.pdf): “The scientific objective of the mission is to expand the lunar scientific knowledge”
  - [ISSDC_CY2_OVERVIEW ISSDC_CY2_OVERVIEW::c00011](https://www.issdc.gov.in/chandrayaan2.html): “science payloads aim to perform detailed study of lunar topography”

### Chandrayaan-3

- Developer finding: **A: no explicit payload developer attribution found in the collected mission-detail page, 12-page brochure, and overview text.**
- Objective finding: **B: the mission objective was already curated; selected payload objectives from the collected details page are now curated with chunk-level provenance.**
- Extraction: The collected overview, details, and brochure were text-extracted without recorded empty pages or warnings.
- Assessment: Mission details explicitly enumerate payloads and objectives and call the lander/propulsion module/rover indigenous, but do not attribute individual payload development to named centres. General ISRO ownership or project participation is not a payload-specific developer edge.
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00002](https://www.isro.gov.in/Chandrayaan3_Details.html): “demonstrate end-to-end capability in safe landing and roving on the lunar surface”
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00023](https://www.isro.gov.in/Chandrayaan3_Details.html): “Objectives: To carry out the measurements of thermal properties”
  - [ISRO_CY3_DETAILS ISRO_CY3_DETAILS::c00024](https://www.isro.gov.in/Chandrayaan3_Details.html): “Objectives: To measure seismicity around the landing site”


Summary: explicit payload objective and selected observation facts from CY1/2/3 are now curated from the existing corpus; CY1 C1XS/SARA developer statements were also curated as written. Some institution/agency relationships remain too ambiguous for developer edges. No explicit payload-specific CY2/CY3 developer attribution was found in reviewed extracted text; that is a corpus gap, not evidence that such attribution does not exist elsewhere.

## 9. Additional official sources and access limits

No additional source was verified sufficiently to enter the registry or be downloaded during this audit. Attempted official ISRO/ISSDC site-search routes returned 404; other search endpoints did not yield inspectable primary documents. No search snippet is treated as evidence.

No extra document is required for the two-hop objective/observation capacity established by this curation. If temporal questions remain in scope, the unresolved collection target is:

| Priority | Candidate source/document family | Organization / mission | Document type and relation gap | Potential new entities, relations, and paths | Candidate URL/title status | Why the current corpus is insufficient |
|---|---|---|---|---|---|---|
| 1 | Official mission-history/sequence document explicitly identifying multiple predecessor/follow-on relations | ISRO/ISSDC; multiple missions | Mission history; `PRECEDED_BY` | Likely no new mission entities; number of new temporal relations and derived multi-hop continuations unknown until an opened primary source is verified. | No additional source verified; no URL/title can be responsibly supplied until inspected. | Only one explicit mission-sequence edge is currently supported. Chronology inferred solely from launch dates is excluded. |
| 2 | Payload-specific development/realization sources naming responsible centres/institutions | ISRO/ISSDC; Chandrayaan-2 and Chandrayaan-3 | Payload documentation; `Payload DEVELOPED_BY Organization/ISROCentre` | Potential new organizations/centres and developer edges; new mission→payload→developer paths depend on specific statements and are not estimated without a source. | Official document/title/URL not yet verified; do not download until opened and exact language checked. | Existing reviewed documents support identity/objectives but not payload-specific developer attribution; current developer paths across four other missions already support a bounded question family. |

### In-corpus curation completed in Phase 1.6

The targeted evidence review also found high-value directly supported claims already inside collected documents:

- Chandrayaan-1: ISSDC payload PDF (`ISSDC_CY1_PAYLOADS::c00001`) explicitly says C1XS and SARA were developed jointly by ESA and ISRO; selected payload objectives and observations were also curated. “From institution through agency” is not automatically a developer claim.
- Chandrayaan-2: ISSDC science-results volume (`ISSDC_CY2_SCIENCE_RESULTS::c00035` and `::c00036`) states mission/science objectives; selected payload objectives and observations are now curated from collected payload documentation.
- Chandrayaan-3: ISRO details page (`ISRO_CY3_DETAILS::c00002` and payload-objective table rows) states the mission objective and payload objectives; selected payload objective/observation facts are now curated.

Thus Phase 1.6 has completed the in-corpus curation. No document was added. Priority 1 temporal-source collection is warranted only if temporal QA remains in scope and should stop if no directly explicit authoritative records are found.

## 10. Final rationale and boundaries

The corpus remains Tier 1 authoritative and all curated triples have chunk-level provenance. Phase 1.6 increases the graph from 98 to 134 triples without adding documents, and adds multi-hop question capacity across payload/developer, payload/observation, and payload/objective patterns. The graph still has 1 explicit temporal edge(s) and 8 directed three-hop paths. Consequently **TARGETED COLLECTION REQUIRED** only if temporal QA is retained as a major intended category; otherwise, collection can stop for a bounded two-hop-focused experiment, with temporal questions excluded and Chandrayaan-2/3 developer attribution reported as a known coverage limitation.

No benchmark, benchmark answer, result file, prior experimental report, or paper was modified or run as part of this audit.
