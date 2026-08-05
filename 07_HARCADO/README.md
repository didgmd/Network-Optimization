# HARCADO

Open-source implementation accompanying the paper:

**Handover-Risk-Cue-Assisted Discrete A3 Control for Low-Mobility Multi-Cell Networks**

## Package structure

```text
src/                    Core HARCADO runtime
├── main.py             Runtime entry point
├── catboost_model.cbm  Released cue model artifact
tools/                  Optional dataset/model preparation tools
simulator/              ns-3 scenario source
ns3_dataset/            Released trajectory dataset
figure_source_data/     Manuscript figure source data
```

## Runtime components

The `src/` directory contains the minimum execution package:

- discrete A3 control environment;
- handover-risk cue interface;
- CatBoost cue model interface;
- DQN/DDQN/D3QN/PPO-based controllers;
- reward and metric contracts.

## Dataset

- `ns3_dataset/20250424_trajectory_all/` contains released ns-3 trajectory and radio-measurement CSV outputs.
- `figure_source_data/` contains source data used to generate manuscript figures.

## Reproducibility

The package is organized to reproduce the reported simulation workflow using the released trajectory data and model artifact. The original ns-3 scenario source is provided under `simulator/`.

## Integrity

`MD5SUMS.txt` records checksums for released data artifacts.

## License

- Source code is released under the MIT License (`LICENSE`).
- Released datasets and figure-source artifacts are released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).

When reusing this package, please preserve the corresponding attribution information and cite the associated manuscript.

## Cite Our Work

If you use this repository in academic research, please cite the associated paper.

Please use the following BibTeX entry:

```bibtex
% The BibTeX entry will be added after the paper is formally published.
```
