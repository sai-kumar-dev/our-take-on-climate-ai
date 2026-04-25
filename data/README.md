# Public Data Package

This directory contains the public data artifact for release `v1.0.0-public-artifact`.

The shipped `sample_dataset.csv` is a representative reproducibility subset of the final ML-ready table. It is a demo artifact for artifact review, schema validation, tests, and example inference payloads. It is not the full raw source bundle, not the full processed training table, and not a claim that complete empirical reproducibility is possible from the subset alone.

## Package Contents

| Path | Purpose |
| --- | --- |
| `sample_dataset.csv` | Representative reproducibility subset of the final ML-ready table. |
| `sample_inputs/` | Example prediction payloads derived from representative rows. |
| `data_dictionary.csv` | Column-by-column field reference. |
| `sample_dataset_schema.json` | Machine-readable schema plus coverage and quality metadata. |
| `DATA_CARD.md` | Dataset card for release and review. |
| `SOURCE_PROVENANCE.md` | Upstream source and licensing notes. |
| `RECONSTRUCTION_GUIDE.md` | Documented reconstruction pathway for local rebuilds. |
| `NOTICE_DATA_SUBSET.md` | Redistribution and licensing boundary notice for the subset. |
| `DATASET_CITATION.txt` | Recommended project and upstream citation guidance. |

Root-level release files that apply to this package:

- `MANIFEST.yaml`
- `dataset_version.json`
- `CHECKSUMS.txt`

## Snapshot

Full final table used to build the package:

- Rows: `24,120`
- Columns: `75`
- States: `26`
- Districts: `480`
- Monthly time steps: `54`
- Time span: `2018-07` to `2022-12`
- Target columns: `41`

Released subset:

- Rows: `211`
- Columns: `75`
- States: `26`
- Districts: `133`
- Monthly time steps: `44`
- Time span: `2018-07` to `2022-11`

## Coverage Summary

| Metric | Released subset | Full final table | Interpretation |
| --- | ---: | ---: | --- |
| States | `26` | `26` | All states are represented. |
| State-season groups | `78` | `78` | All `state x season` groups are represented. |
| Regions | `133` | `480` | Geographic coverage is intentionally partial. |
| Months | `44` | `54` | Temporal coverage is intentionally partial. |
| Dominant crops | `23` | `23` | All crops that are the row-wise top target somewhere in the full table appear as dominant at least once in the subset. |
| Target columns positive at least once | `41` | `41` | Every `crop_prob_*` column has at least one positive exemplar. |

Important distinctions:

- `41/41` positive target coverage is not the same thing as dominant-crop coverage.
- `23/23` dominant-crop coverage is the stricter argmax-style crop coverage metric.
- `133/480` region coverage and `44/54` month coverage are partial by design and should be described that way.

## Quality Variation Actually Represented

The subset preserves some, but not all, quality variation dimensions from the full table.

- `soil_imputed`: both `0` and `1` are present
- `data_confidence`: `0.9375` to `1.0`
- `target_season`: `kharif`, `rabi`, and `zaid`
- `geo_confidence`: only `1.0` appears in the released subset
- `climate_gap_filled`: only `0` appears in the released subset
- `time_step_missing`: only `0` appears in the released subset

## Sampling Strategy

The subset is deterministic and coverage-driven rather than random.

It is built as the union of three anchor rules:

1. Per `state x season`, include the lowest- and highest-rainfall rows with quality-aware tie-breaking.
2. For every `crop_prob_*` target, include the row with the maximum observed value.
3. For every state, include the lowest-confidence row to preserve nontrivial quality variation.

After deduplication, the final subset is sorted by `state`, `region`, and `time`.

## Scope Decision

No additional anchor rows were added for this public release. Keeping the subset at 211 rows preserves all required state, season, dominant-crop, and target-column coverage while keeping the artifact clearly bounded as a representative demo artifact rather than a near-complete national slice.

## Reconstruction Boundary

This release publishes:

- a schema-compatible subset
- documentation and provenance notes
- a documented reconstruction pathway
- release fixity metadata

This release does not publish:

- the full raw source bundle
- the full processed training table
- every intermediate internal ETL artifact

Use [RECONSTRUCTION_GUIDE.md](RECONSTRUCTION_GUIDE.md) for the honest rebuild path.

## Licensing And Citation Boundary

Reviewer-safe wording:

> This representative subset is provided for reproducibility demonstration. Users should consult original upstream sources for full source licensing.

The Apache-2.0 code license at the repository root does not relicense third-party upstream data. Use [NOTICE_DATA_SUBSET.md](NOTICE_DATA_SUBSET.md) and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) as the source of truth for release-facing data wording.
