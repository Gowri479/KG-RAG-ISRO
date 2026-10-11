# Graph-Based Baselines Comparative Evaluation Report

## Architectural Differences Across Systems

- **BM25 + LLM:** Sparse lexical retrieval over raw document chunks (BM25Okapi); no semantic vectors or graph structure.
- **Vanilla RAG:** Dense vector search over raw document chunks (MiniLM-L6-v2, top-3 FAISS); no relational graph.
- **GraphRAG-style baseline:** Greedy modularity community detection over the domain KG. Retrieves the most semantically relevant community summary.
- **LightRAG-inspired baseline:** Dual-level retrieval combining query-entity 1-hop KG subgraphs with community overview summaries.
- **KG-RAG (ours):** Integrated hybrid retrieval joining query keyword filtering, domain entity 1-hop relational triples, and dense FAISS text passages.

## Performance Comparison on Targeted Aditya-L1 Benchmark (36 Questions)

| System | Architecture | ROUGE-L | Coverage | Exact Match | IDK% | Mean Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GraphRAG-style baseline** | Modularity community detection + summary embedding retrieval | 0.0487 | 0.0860 | 0.0000 | 72.2% | 9826.1 |
| **LightRAG-inspired baseline** | Dual-level retrieval (local 1-hop triples + modularity community context) | 0.0821 | 0.1034 | 0.0000 | 77.8% | 14871.1 |
| **BM25 + LLM** | Sparse BM25 passage retrieval over unstructured chunks | 0.2918 | 0.4767 | 0.0278 | 22.2% | 1150.0 |
| **Vanilla RAG** | MiniLM dense passage retrieval over unstructured chunks | 0.2924 | 0.4897 | 0.0556 | 25.0% | 1280.0 |
| **KG-RAG (ours)** | Hybrid: Keyword pre-filter + 1-hop KG neighborhood triples + FAISS passages | 0.3859 | 0.5473 | 0.0556 | 8.3% | 1420.0 |
