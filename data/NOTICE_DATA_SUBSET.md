# NOTICE_DATA_SUBSET

This notice applies to the public data artifact under `data/`.

The released `sample_dataset.csv` is a derived subset of the final ML-ready table used by the project. It is released as a representative reproducibility subset and demo artifact for schema inspection, artifact review, tests, and example inference payloads.

Reviewer-safe statement:

> This representative subset is provided for reproducibility demonstration. Users should consult original upstream sources for full source licensing.

Important boundary:

- The repository Apache-2.0 license covers original code and repository-authored documentation.
- The Apache-2.0 code license does not grant broader rights in third-party upstream source data.
- The public subset should not be described as a relicensed copy of all upstream raw data.
- Upstream licensing must be evaluated at the original source level as documented in `SOURCE_PROVENANCE.md`.

Release-facing guidance:

- Cite the project artifact.
- Cite upstream sources separately.
- Do not imply that the repo ships the full raw source bundle or full processed training data.
