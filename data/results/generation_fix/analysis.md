# Generation-fix analysis

## Scope and integrity

- 60 KG-required questions and 180 A/B/C rows were verified.
- Retrieval was reused exactly from the frozen Relation-Aware KG-RAG traces; no retrieval experiment was rerun.
- Frozen benchmark SHA-256: `6c7f3600995c0091de73d20d2f02953cb414023df281f7178069f9f81b432b64`
- Frozen prior-generation SHA-256: `e13e4c82a3638df15832887621a066b57afea62da5ce8d28394cb18cd608635c`

## Overall automated results

| Condition | ROUGE-L | Coverage | Exact match | IDK |
|---|---:|---:|---:|---:|
| A_CURRENT | 0.148 | 0.360 | 0.000 | 0.017 |
| B_STRUCTURED | 0.138 | 0.416 | 0.000 | 0.000 |
| C_TWO_STAGE | 0.217 | 0.507 | 0.000 | 0.100 |

## Category results

| Category | N | A ROUGE-L | B ROUGE-L | C ROUGE-L | A coverage | B coverage | C coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_CONTROL | 10 | 0.228 | 0.150 | 0.320 | 0.367 | 0.383 | 0.583 |
| MULTI_RELATION | 10 | 0.109 | 0.147 | 0.097 | 0.211 | 0.264 | 0.166 |
| SINGLE_RELATION | 30 | 0.171 | 0.158 | 0.269 | 0.391 | 0.426 | 0.548 |
| TWO_HOP_RELATION | 10 | 0.034 | 0.057 | 0.074 | 0.408 | 0.567 | 0.650 |

## Failure analysis

- Failures where B or C lost on at least one lexical metric: 29
- Categories: `{'Evidence dilution': 29}`

## Decision

**PARTIAL SUPPORT**

Human evaluation was triggered for consideration because a meaningful automated improvement was observed.

Automated metrics remain lexical diagnostics and do not establish factual quality.
