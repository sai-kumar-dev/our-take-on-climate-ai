# Repository Guide

This guide is for first-time visitors, recruiters, open-source users, and artifact reviewers who want the fastest route through the repository.

## Recommended Reading Order

1. Start with [README.md](../README.md) for the public artifact summary and quickstart.
2. Open [data/README.md](../data/README.md) to understand the released data package.
3. Review [data/DATA_CARD.md](../data/DATA_CARD.md) and [data/SOURCE_PROVENANCE.md](../data/SOURCE_PROVENANCE.md) for release-facing dataset scope and provenance.
4. Read [data/RECONSTRUCTION_GUIDE.md](../data/RECONSTRUCTION_GUIDE.md) for the full rebuild boundary.
5. Inspect [CITATION.cff](../CITATION.cff), [MANIFEST.yaml](../MANIFEST.yaml), [dataset_version.json](../dataset_version.json), and [CHECKSUMS.txt](../CHECKSUMS.txt) for citation and fixity metadata.
6. Use [PROJECT_GUIDE.md](../PROJECT_GUIDE.md) and [docs/ARCHITECTURE.md](ARCHITECTURE.md) for deeper engineering context.

## Repository Map

| Path | What it contains | Open this if you want |
| --- | --- | --- |
| [`README.md`](../README.md) | Landing page, quickstart, and public release overview | The fastest overall orientation |
| [`data/`](../data) | Public data artifact and supporting documentation | The shipped dataset and reviewer-facing artifact materials |
| [`data/sample_dataset.csv`](../data/sample_dataset.csv) | Representative reproducibility subset | The actual released sample table |
| [`data/DATA_CARD.md`](../data/DATA_CARD.md) | Dataset card | Scope, intended use, and limitations |
| [`data/SOURCE_PROVENANCE.md`](../data/SOURCE_PROVENANCE.md) | Upstream provenance notes | Source lineage and licensing context |
| [`data/RECONSTRUCTION_GUIDE.md`](../data/RECONSTRUCTION_GUIDE.md) | Rebuild instructions | How to reconstruct the full local pipeline inputs |
| [`src/`](../src) | Application and ML pipeline source code | Implementation details |
| [`docs/`](../docs) | Supporting technical docs | Architecture and governance notes |
| [`PROJECT_GUIDE.md`](../PROJECT_GUIDE.md) | Engineering walkthrough | A deeper technical read than the README |
| [`tests/`](../tests) | Unit and integration tests | What is covered and how behavior is verified |
| [`configs/`](../configs) | Example and training configs | Pipeline and training configuration surfaces |

## Release Metadata

These files are the tracked release-facing metadata surface for the public artifact:

- [`CITATION.cff`](../CITATION.cff)
- [`MANIFEST.yaml`](../MANIFEST.yaml)
- [`dataset_version.json`](../dataset_version.json)
- [`CHECKSUMS.txt`](../CHECKSUMS.txt)
- [`LICENSE`](../LICENSE)
- [`NOTICE`](../NOTICE)

## Generated Vs. Tracked Paths

The repository intentionally separates tracked public-release materials from generated local outputs.

- Tracked reviewer-facing materials live in `README.md`, `data/`, `docs/`, and the root release metadata files.
- Local generated outputs such as `artifacts/` and `data/processed/` are not versioned in git by default.
- If you run training or evaluation locally, generated reports and plots are written under `artifacts/`.

## Reviewer-Focused Path

If you are reviewing the public research artifact, the shortest path is:

1. [README.md](../README.md)
2. [data/README.md](../data/README.md)
3. [data/sample_dataset.csv](../data/sample_dataset.csv)
4. [data/SOURCE_PROVENANCE.md](../data/SOURCE_PROVENANCE.md)
5. [data/RECONSTRUCTION_GUIDE.md](../data/RECONSTRUCTION_GUIDE.md)
6. [CITATION.cff](../CITATION.cff), [MANIFEST.yaml](../MANIFEST.yaml), and [CHECKSUMS.txt](../CHECKSUMS.txt)

## Local Evaluation Outputs

Evaluation outputs are written to local `artifacts/evaluation/` after running:

```bash
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

That path is useful for local review, but it is a generated output directory rather than a tracked documentation surface.
