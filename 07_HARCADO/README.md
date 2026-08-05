# HARCADO

Open-source implementation accompanying the paper:

**Handover-Risk-Cue-Assisted Discrete A3 Control for Low-Mobility Multi-Cell Networks**

## Package structure

```text
src/                    Core HARCADO runtime
 tools/                 Optional dataset/model preparation tools
 simulator/             ns-3 scenario source
 ns3_dataset/           Released trajectory dataset
 figure_source_data/    Manuscript figure source data
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
- `figure_source_data/` contains source data used for manuscript figures.

## Reproducibility

The released package is intended to reproduce the reported simulation workflow using the provided dataset and model artifacts.

## Integrity

`MD5SUMS.txt` records checksums for released data artifacts.

## License

Unless otherwise noted, released datasets and source files are provided under the selected open-source licenses. Users should cite the associated manuscript when reusing this package.
