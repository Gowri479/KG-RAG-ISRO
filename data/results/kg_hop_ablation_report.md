# Knowledge Graph Ablation & Hop Analysis Report

## Evaluated Variants

| Variant | Description | ROUGE-L | Coverage | Exact Match | IDK% | Mean Context Words | Mean Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **dense_only** | Dense Only | 0.2924 | 0.4897 | 0.0000 | 25.0% | 293 | 1200.0 |
| **kg_only** | Kg Only | 0.0553 | 0.0852 | 0.0000 | 75.0% | 30 | 7837.5 |
| **dense_kg_1hop** | Dense Kg 1Hop | 0.3353 | 0.5444 | 0.0000 | 11.1% | 821 | 58636.8 |
| **dense_kg_2hop** | Dense Kg 2Hop | 0.2926 | 0.5096 | 0.0000 | 19.4% | 822 | 16709.3 |
| **bm25_only** | Bm25 Only | 0.2918 | 0.4767 | 0.0000 | 22.2% | 300 | 1200.0 |
| **bm25_kg** | Bm25 Kg | 0.3511 | 0.6158 | 0.0000 | 13.9% | 1252 | 84632.5 |
| **full_kg_rag** | Full Kg Rag | 0.3859 | 0.5473 | 0.0000 | 8.3% | 296 | 1200.0 |

## Paired Statistical Testing

| Comparison | Mean Diff (ROUGE-L) | 95% Bootstrap CI | Wilcoxon p | Mean Diff (Cov) | 95% Bootstrap CI | Wilcoxon p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **dense_kg_1hop_vs_dense_only** | +0.0430 | [-0.0281, 0.1141] | 0.1229 | +0.0548 | [-0.0366, 0.1572] | 0.4488 |
| **dense_kg_2hop_vs_dense_kg_1hop** | -0.0428 | [-0.1147, 0.0283] | 0.2942 | -0.0348 | [-0.1187, 0.0520] | 0.5034 |
| **bm25_kg_vs_bm25_only** | +0.0594 | [-0.0416, 0.1622] | 0.1421 | +0.1391 | [-0.0029, 0.2818] | 0.0694 |
| **full_kg_rag_vs_dense_only** | +0.0935 | [0.0297, 0.1674] | 0.0067 | +0.0577 | [-0.0478, 0.1722] | 0.3203 |
| **full_kg_rag_vs_dense_kg_1hop** | +0.0505 | [-0.0241, 0.1243] | 0.1180 | +0.0029 | [-0.0634, 0.0678] | 0.7323 |
