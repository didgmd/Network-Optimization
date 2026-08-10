# Adaptive UE Handover Management with MAR-Aided Multivariate DQN in Ultra-Dense Networks

Open-source archival package accompanying the paper:

**Adaptive UE Handover Management with MAR-Aided Multivariate DQN in Ultra-Dense Networks**

> This release is an archived/reference implementation package. Due to unavailable historical execution snapshots, the released runtime should not be interpreted as the exact code snapshot used to generate every reported manuscript result.

## Package structure

```text
src/                    Archived runtime implementation
figure_source_data/     Manuscript figure-source data
```

## Runtime components

The `src/` directory preserves the available Python implementation from the associated research project, including:

- ultra-dense network topology generation;
- Lévy Walk user mobility model;
- three DQN-based handover modules:
  - handover decision;
  - A2/A4 threshold adjustment;
  - target base-station selection;
- Memory Anchor Repository (MAR) mechanism;
- KPI evaluation utilities.

## Reproducibility scope

The package provides the available implementation artifacts and figure-source data associated with the study.

Because the exact historical experiment snapshot is unavailable, this release is intended for archival reference and code inspection rather than a claim of exact manuscript-result reproduction.

## Figure source data

The `figure_source_data/` directory contains the source data corresponding to manuscript evaluation figures.

## Requirements

Python dependencies are listed in `requirements.txt`.

## License

- Source code: MIT License (`LICENSE`).
- Figure-source data: CC BY 4.0 (`LICENSE-DATA`).

Please preserve attribution and cite the associated manuscript when reusing this package.

## Cite Our Work

```bibtex
% BibTeX entry placeholder
```
