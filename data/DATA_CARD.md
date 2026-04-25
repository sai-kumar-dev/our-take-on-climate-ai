# DATA_CARD

## Dataset Name

Representative Reproducibility Subset

## Release Summary

This package supports a public research and software release without claiming that the repository redistributes the full raw data stack or the full processed training table.

The artifact is best described as:

- a representative reproducibility subset
- a demo artifact
- a schema-faithful public slice of the final ML-ready table

It should not be described as the full training corpus or as complete empirical reproducibility from the subset alone.

## Composition

Full final table snapshot:

- Rows: `24,120`
- Columns: `75`
- States: `26`
- Districts: `480`
- Time span: `2018-07` to `2022-12`
- Target columns: `41`

Public subset snapshot:

- Rows: `211`
- Columns: `75`
- States: `26`
- Districts: `133`
- Time span: `2018-07` to `2022-11`
- Seasons represented: `kharif`, `rabi`, `zaid`

The table contains:

- district and time identifiers
- aggregated climate features
- soil and soil-support features
- contextual and quality metadata
- `crop_prob_*` target columns that sum to `1` row-wise

## Coverage Summary

| Metric | Released subset | Full final table | Meaning |
| --- | ---: | ---: | --- |
| States | `26` | `26` | All states are represented. |
| State-season groups | `78` | `78` | All `state x season` groups are represented. |
| Regions | `133` | `480` | Region coverage is partial. |
| Months covered | `44` | `54` | Month coverage is partial. |
| Dominant crops | `23` | `23` | All crops that are row-wise dominant anywhere in the full table appear as dominant at least once in the subset. |
| Target columns positive at least once | `41` | `41` | Every target column has at least one positive exemplar. |

Interpretation:

- Target-column positive coverage and dominant-crop coverage are different metrics.
- Region and month coverage are intentionally partial and should be described that way.
- The subset is built for coverage and reproducibility demonstration, not for national statistical estimation.

## Sampling Method

The subset is deterministic and coverage-driven rather than random.

Rows were selected by taking the union of:

1. Per `state x season` rainfall extremes.
2. Per-target maximum-probability exemplars across all `crop_prob_*` columns.
3. Per-state lowest-confidence rows to retain nontrivial quality variation.

This makes the subset useful for schema and behavior coverage, but not a random sample of the national distribution.

## Quality Variation Actually Represented

The released subset preserves only the following quality variation dimensions:

- `soil_imputed`: both `0` and `1`
- `data_confidence`: `0.9375` to `1.0`
- `target_season`: `kharif`, `rabi`, and `zaid`

The following dimensions are not varied inside the public subset:

- `geo_confidence`: only `1.0`
- `climate_gap_filled`: only `0`
- `time_step_missing`: only `0`

## Target Semantics

The `crop_prob_*` columns are normalized district-time probability targets derived from historical crop records. They are best described as:

- historical cultivation distribution targets
- regional pattern proxies

They should not be described as:

- ground-truth agronomic optima
- direct yield labels
- causal outcome variables

## Intended Use

Appropriate uses:

- public repository demo data
- artifact review
- schema validation
- example API payloads
- lightweight tests and documentation examples

Not appropriate as:

- the full training dataset
- a production agronomic decision dataset
- a standalone benchmark corpus
- a substitute for source-by-source provenance review

## Scope Decision

No extra anchor rows were added for `v1.0.0-public-artifact`. The current sample already covers all states, all state-season groups, all dominant crops, and all target columns while remaining clearly bounded as a public demo artifact.

## Known Limitations

- The subset is coverage-driven and curated, not random.
- `N`, `P`, and `K` are proxy-style district features in this build, not direct laboratory observations for every row.
- `soil_health_index` is a simple internal composite feature, not an official soil health card metric.
- `irrigation_index` and `rotation_score` are heuristic management-context features in this build.
- The subset does not cover all regions or all months in the full table.
- The subset alone does not support complete empirical reconstruction of the full study.

## Licensing And Citation Notes

Reviewer-safe wording:

> This representative subset is provided for reproducibility demonstration. Users should consult original upstream sources for full source licensing.

Additional guidance:

- Treat the package as a derived research artifact.
- Cite the project and cite upstream sources separately.
- Do not imply that the Apache-2.0 repository license relicenses third-party upstream data.
