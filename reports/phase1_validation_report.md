# Phase 1 structural validation

- Status: **PASSED**
- Checks passed: 19/19
- Registered documents: 33
- Chunks: 1006
- Entities: 123
- Triples with provenance: 134/134

This is structural validation; live HTTP reachability of source URLs was not tested by this script.

| Check | Result | Detail |
|---|---|---|
| unique_registry_document_ids | PASS | 33 registered documents |
| registry_required_metadata | PASS | Missing required metadata: [] |
| official_source_urls_well_formed | PASS | Each URL is HTTPS and its host matches source_domain. |
| registered_documents_are_tier1 | PASS | Only official ISRO/ISSDC primary sources are in this corpus. |
| registered_source_files_exist | PASS | Missing source files: [] |
| registered_source_hashes_match | PASS | Checksum mismatches: [] |
| no_duplicate_document_hashes | PASS | Duplicate hashes: {} |
| every_raw_file_is_registered | PASS | Unregistered raw files: [] |
| unique_chunk_ids | PASS | 1006 chunks |
| chunks_map_to_registered_documents | PASS | Unregistered chunk references: [] |
| chunk_source_urls_match_registry | PASS | Mismatched chunk URLs: [] |
| chunks_have_text_section_and_tier | PASS | Incomplete chunks: [] |
| unique_entity_ids | PASS | 123 entities |
| entities_use_ontology_types | PASS | Invalid entity types: [] |
| unique_triple_ids | PASS | 134 triples |
| triples_are_typed_controlled_and_unique | PASS |  |
| no_orphan_entities | PASS | Orphan entities: [] |
| every_entity_has_chunk_provenance | PASS |  |
| every_triple_has_exact_chunk_provenance | PASS |  |
