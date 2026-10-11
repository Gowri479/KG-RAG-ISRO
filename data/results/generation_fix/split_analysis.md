# Split-aware generation-fix analysis

The existing 180 generations were reused without regeneration. Condition C is unchanged.

## Primary held-out evaluation

- Questions: 13
- IDs: `rakg_005, rakg_010, rakg_015, rakg_020, rakg_025, rakg_030, rakg_035, rakg_040, rakg_050, rakg_055, rakg_060, rakg_065, rakg_070`

| Condition | ROUGE-L | Coverage | Exact match | IDK |
|---|---:|---:|---:|---:|
| A_CURRENT | 0.104 | 0.223 | 0.000 | 0.000 |
| B_STRUCTURED | 0.070 | 0.176 | 0.000 | 0.000 |
| C_TWO_STAGE | 0.233 | 0.412 | 0.000 | 0.077 |

### C_TWO_STAGE minus A_CURRENT on held-out evaluation

- ROUGE-L mean difference: `0.129947`
- ROUGE-L median difference: `0.004091`
- ROUGE-L bootstrap 95% CI: `[0.004757823076923078, 0.2699059384615384]`
- ROUGE-L Wilcoxon p-value: `0.09765625`
- Coverage mean difference: `0.188462`
- Coverage median difference: `0.000000`
- Coverage bootstrap 95% CI: `[0.028205076923076926, 0.3794872307692308]`
- Coverage Wilcoxon p-value: `0.125`
- Coverage wins/losses/ties: `4/1/8`

The held-out result is decisive. The all-60 improvement does not by itself establish generalization.

## Development results

- Questions: 47

| Condition | ROUGE-L | Coverage | Exact match | IDK |
|---|---:|---:|---:|---:|
| A_CURRENT | 0.160 | 0.398 | 0.000 | 0.021 |
| B_STRUCTURED | 0.157 | 0.482 | 0.000 | 0.000 |
| C_TWO_STAGE | 0.212 | 0.534 | 0.000 | 0.106 |

## Conclusion

The 13-question held-out evaluation, not the 60-question aggregate, determines whether C_TWO_STAGE beats A_CURRENT. Automated metrics do not establish factual quality; human evaluation is required for that claim.
