# RECONSTRUCTION_GUIDE

## Purpose

This guide defines the public reconstruction boundary for `v1.0.0-public-artifact`.

The repository publishes:

- a representative reproducibility subset
- schema and data dictionary files
- provenance and licensing notes
- a documented reconstruction pathway

The repository does not publish the full raw source bundle or the full processed training table.

The subset alone is therefore useful for demos, artifact review, and interface inspection, but it is not a claim of complete empirical reproducibility from the shipped rows alone.

## What The Public Release Includes

- `data/sample_dataset.csv`
- `data/sample_inputs/*.json`
- `data/data_dictionary.csv`
- `data/sample_dataset_schema.json`
- `data/DATA_CARD.md`
- `data/SOURCE_PROVENANCE.md`
- `data/NOTICE_DATA_SUBSET.md`
- `data/DATASET_CITATION.txt`
- `MANIFEST.yaml`
- `dataset_version.json`
- `CHECKSUMS.txt`

## What The Public Release Does Not Include

- the full raw upstream source bundle
- the full 24,120-row processed training table
- every intermediate ETL notebook, scratch stage, or internal builder artifact
- enough rows to reproduce all empirical coverage and evaluation behavior from the subset alone

That omission is intentional and should be described honestly in release notes and artifact appendices.

## Documented Reconstruction Pathway

If you need the full final-table interface rather than only the public subset, use the following high-level path:

1. Acquire the documented upstream sources listed in `SOURCE_PROVENANCE.md`.
2. Harmonize district and state names into stable region keys.
3. Build the three prepared source tables expected by the repository:
   - `climate_daily.csv`
   - `soil_samples.csv`
   - `crop_monthly.csv`
4. Place those prepared tables under `data/raw/data_new/`.
5. Run the pipeline:

```bash
python run_pipeline.py --config configs/data_new_config.json
```

6. This produces the full final table locally at:

```text
data/processed/data_new_final_ml_dataset.csv
```

7. If you need model artifacts and evaluation outputs, continue with:

```bash
python train_model.py --config configs/training_data_new.json
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

8. If the full local table is present, regenerate the public package and release metadata with:

```bash
python scripts/build_public_data_package.py
```

## Prepared Source Table Interfaces

The repository-level pipeline expects these canonical prepared tables.

### `climate_daily.csv`

Expected fields:

- `region`
- `state`
- `date`
- `temperature`
- `rainfall`
- `humidity`

### `soil_samples.csv`

Expected fields:

- `region`
- `state`
- `ph`
- `n_kg_ha`
- `p_kg_ha`
- `k_kg_ha`

Important note:

In this project build, downstream `N`, `P`, and `K` features are documented as proxy-style district features rather than universally direct raw lab measurements for every final row.

### `crop_monthly.csv`

Expected fields:

- `region`
- `state`
- `year`
- `month`
- `crop`
- `area`
- `production`

## Public Wording To Use

Recommended wording:

- "The repository includes a representative reproducibility subset of the final ML-ready dataset."
- "The public release is a demo artifact plus a documented reconstruction pathway."
- "The full final table can be rebuilt locally from documented upstream sources and the repository pipeline interface."

Avoid wording like:

- "The full raw data ships in the repo."
- "The full processed training data is included."
- "Complete empirical reproducibility is available from the subset alone."

## Licensing Reminder

Use the following reviewer-safe statement when describing the data release:

> This representative subset is provided for reproducibility demonstration. Users should consult original upstream sources for full source licensing.

The Apache-2.0 repository license applies to original code and repository-authored documentation. It does not grant broader rights in third-party source data.
