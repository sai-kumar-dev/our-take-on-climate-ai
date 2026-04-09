# Model Governance

## Promotion Checklist

Before promoting a model artifact:

1. Review test metrics and calibration.
2. Review slice metrics for state, season, and month.
3. Review geography overlap metrics.
4. Review sanity scenario behavior.
5. Confirm the model card and run manifest are present.
6. Confirm the target definition still matches the product wording.

## Current Risk Classification

- Product class: advisory shortlist guidance
- User sensitivity: medium to high
- Human review needed for training feedback reuse: yes
- Human review needed before public farmer deployment: yes

## Registry Files

- `artifacts/registry/training_runs.jsonl`
- `artifacts/registry/model_registry.jsonl`

These files are append-only local records for run lineage and candidate promotion tracking.
