# Phase 1 corpus gap analysis

## Gap assessment and targeted follow-up

The collection pass prioritized the seven specified mission programmes, and the follow-up collection added mission launch pages and official payload/science publications where initial coverage was thin. This is a bounded collection, not an exhaustive catalogue of ISRO publications.

1. **Payload-developer chains vary by mission.** Aditya-L1 has explicit organization attribution for its seven payloads, and AstroSat's five payload developer attributions are explicit in its mission page. Mars Orbiter Mission has a directly stated developer for TIS. The collected Chandrayaan-1/2/3 sources currently support payload membership, but no developer edges were curated where the source statements did not establish the attribution directly. This leaves uneven multi-hop coverage and is not filled by centre co-occurrence or presumed institutional roles.
2. **Temporal relation coverage is sparse by design.** Chandrayaan-3 is directly described as a follow-on to Chandrayaan-2. No other predecessor edge is asserted from chronology alone. Additional sequence claims require explicit primary-source wording.
3. **Gaganyaan is a programme under development, not a completed crewed launch.** The official source describes the proposed mission and identifies HSFC as lead centre. No actual launch date, launch event, or flown payload relation is asserted.
4. **Science and objective links are selective.** The KG-ready facts preserve direct mission-level targets/objectives where explicit, but payload objectives are not exhaustively converted to triples. The source documents and chunks retain broader science evidence for later reviewed annotation.
5. **Extraction limitations remain visible.** The Chandrayaan-1 brochure is scanned/image-only and produced no text layer; it remains registered and raw-only. One page in the Chandrayaan-2 brochure also produced no text. Extraction flagged 17 PDF pages with private-use glyphs, retaining the text unchanged for review. No OCR-derived claims were introduced.
6. **Source-access limitations are explicit.** Two legacy PRADAN/ISSDC portal URLs recorded in the pre-existing Aditya pilot manifest returned HTTP 504, and some older ISRO routes returned 404. They were not included as evidence; the selected corpus uses the valid current official routes in the registry. Structural validation checks local checksums and URL form but does not make live network requests.
7. **Coverage is not uniform by volume.** Chandrayaan-2 has far more chunks because its large official science-results publication and payload-data handbook were included. Chunk counts should not be interpreted as answer coverage or comparative evidence quality.

## Stop criterion

The second search pass improved launch and payload/science source coverage. Remaining omissions are either facts not directly established in the collected text or deliberately deferred to avoid inferred relations and unbounded collection. The corpus is suitable as a Phase 1 provenance-preserving, KG-ready initial resource; it is not claimed to be exhaustive. Further source collection should be a separately recorded, evidence-led extension rather than driven by benchmark performance.
