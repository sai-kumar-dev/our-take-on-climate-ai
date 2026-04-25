# SOURCE_PROVENANCE

This document records the upstream sources referenced by the project and clarifies how they relate to the **Representative Reproducibility Subset**.

Important boundary:

- The public package redistributes only a derived subset of the **final ML-ready table**.
- It does **not** redistribute the full raw source bundle or the full evolving internal extraction pipeline.
- Some provenance details come from repository documentation drafts and local preparation reports rather than from a complete published ETL release.
- The Apache-2.0 repository license applies to original code and repository-authored documentation. It does **not** grant broader rights in third-party upstream source data.
- Reviewer-safe release note: "This representative subset is provided for reproducibility demonstration. Users should consult original upstream sources for full source licensing."

## Prepared Internal Interface

The internal pipeline expects three prepared source tables:

- `data/raw/data_new/climate_daily.csv`
- `data/raw/data_new/soil_samples.csv`
- `data/raw/data_new/crop_monthly.csv`

Those prepared tables already reflect upstream integration work. The public subset is derived **after** those tables have been processed and merged.

## Upstream Source Summary

| Source | Official URL | Variables / role in project | Evidence in repo | Licensing notes | Public package treatment |
| --- | --- | --- | --- | --- | --- |
| OGD India daily district-wise rainfall | https://www.data.gov.in/resource/daily-district-wise-rainfall-data | Rainfall inputs used to build `rain_total`, `rain_variance`, `max_rain_1d`, `dry_spell_days`, and rainfall lag features. | `Section_5_1_Data_Sources.txt`, `Section_14_References.txt`, climate feature code in `src/climate_pipeline/transforms.py` | The OGD listing states datasets are licensed under the Government Open Data License - India. Verify attribution terms before redistribution. | Not redistributed directly. Only derived district-month features appear in `sample_dataset.csv`. |
| NASA POWER API | https://power.larc.nasa.gov/docs/services/api/ | Weather and temperature inputs used to build `temp_avg`, `max_temp`, `max_temp_3d`, `humidity_avg`, and temperature lag features. | `Section_5_1_Data_Sources.txt`, `Section_14_References.txt`, climate feature code | NASA Earth science data policy states these data are openly available to users for any purpose; cite the service and follow API guidance. | Not redistributed directly. Only derived district-month features appear in `sample_dataset.csv`. |
| ISRIC SoilGrids | https://docs.isric.org/globaldata/soilgrids/ | Soil pH and other soil properties used to construct district soil support features. | `Section_5_1_Data_Sources.txt`, `Section_14_References.txt`, `reports/data_new_preparation_report.json` | SoilGrids documentation states the maps are publicly available under CC-BY 4.0. | Not redistributed directly. Derived district-level soil features are included in `sample_dataset.csv`. |
| Kaggle: Crop Production in India | https://www.kaggle.com/datasets/abhinand05/crop-production-in-india | Historical crop area / production records used to derive district-time crop probability targets. | `Section_5_1_Data_Sources.txt`, `Section_14_References.txt`, `reports/data_new_preparation_report.json` | The Kaggle page currently reports `License: Other (specified in description)`. Review the original source chain before redistribution. | Not redistributed directly. Only normalized `crop_prob_*` targets are included in the public subset. |
| Directorate of Economics and Statistics crop APY reports | https://data.desagri.gov.in/website/crops-apy-report-web | Additional crop statistics and agricultural context referenced in project documentation. | `Section_5_1_Data_Sources.txt`, `Section_14_References.txt` | Public report pages are visible, but a clear redistribution license was not established from repo evidence. Treat as license-review pending. | Not redistributed directly. Any influence is only through derived final-table fields. |
| Nominatim / OpenStreetMap geocoding | https://operations.osmfoundation.org/policies/nominatim/ | District-to-coordinate mapping and geographic alignment support during source harmonization. | `Section_5_1_Data_Sources.txt` | Nominatim usage policy applies to the public service. OSM data is ODbL; bulk or repeated querying must follow the stated limits. | No raw geocoding results are included in the public subset. |

## Source-To-Feature Mapping

### Climate-derived columns

Primary columns:

- `temp_avg`
- `rain_total`
- `humidity_avg`
- `rain_variance`
- `max_temp`
- `max_temp_3d`
- `max_rain_1d`
- `dry_spell_days`
- `temp_lag_7`
- `rain_lag_14`

Interpretation:

- rainfall-heavy variables are consistent with the OGD rainfall reference
- temperature and humidity variables are consistent with the NASA POWER reference
- the prepared internal file `climate_daily.csv` appears to act as the unified climate interface

### Soil-derived columns

Primary columns:

- `pH`
- `N`
- `P`
- `K`
- `N_class`
- `P_class`
- `K_class`
- `soil_health_index`

Important qualification:

`reports/data_new_preparation_report.json` explicitly states that `N`, `P`, and `K` in the prepared soil table are **proxy estimates** derived from other soil variables because the internal `data_new` soil file does not contain direct NPK measurements.

That means public documentation should avoid phrasing these columns as direct district lab measurements.

### Context and quality columns

Primary columns:

- `irrigation_index`
- `rotation_score`
- `fertility_class`
- `state_context`
- `region_context`
- `target_month`
- `target_season`
- `time_step_missing`
- `climate_gap_filled`
- `soil_imputed`
- `geo_confidence`
- `data_confidence`

Important qualification:

Several of these are internal derived fields rather than upstream observed measurements. In particular:

- `irrigation_index` is a bounded heuristic context feature in this build
- `rotation_score` is a bounded heuristic context feature in this build
- `soil_health_index` is a simple internal composite feature in this build

### Target columns

Primary columns:

- all `crop_prob_*` columns

Interpretation:

- These are normalized district-time probability targets derived from historical crop records.
- They are row-normalized and sum to `1`.
- They represent historical cultivation distribution, not direct agronomic optimum labels.

## Provenance Gaps To Acknowledge Publicly

The following items should be stated plainly in the public release:

- The repo does not publish every intermediate extraction and enrichment stage used during internal data assembly.
- The prepared source files collapse multiple upstream operations into three interface tables.
- Not every upstream redistribution license has been fully verified inside the repository itself.
- The public package is therefore a **derived reproducibility subset**, not a raw-source archive.

## Recommended Citation Practice

When publishing or releasing the package:

1. Cite the project repository or software artifact.
2. Cite the upstream climate, soil, and crop data sources separately.
3. State clearly that `sample_dataset.csv` is a derived subset of the final ML-ready table.
4. Do not imply that the public package contains all original upstream raw data.
5. Do not imply that the Apache-2.0 code license supersedes upstream data terms.
