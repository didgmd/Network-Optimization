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

```bibtex
@article{YUE2026112638,
  title = {Handover-Risk-Cue-Assisted Discrete {A3} Control for Low-Mobility Multi-Cell Networks},
  author = {Yue, Zicheng and Yang, Heng and Li, Shanshan and Feng, Jieqiong and Khan, Navid Ali and Guo, Hongzhang and Kong, Yuwei and Zhang, Lei and Chen, Siqi},
  journal = {Computer Networks},
  volume = {288},
  pages = {112638},
  year = {2026},
  issn = {1389-1286},
  publisher = {Elsevier},
  doi = {10.1016/j.comnet.2026.112638},
  url = {https://www.sciencedirect.com/science/article/pii/S138912862600650X},
  keywords = {A3 handover control, Mobility robustness optimization, Discrete reinforcement learning, Handover-risk cue, Weak-coverage forced handover, Multi-cell network},
  abstract = {Adaptive A3 handover control in multi-cell networks must respond to degrading serving links while avoiding ping-pong instability. This paper presents a handover-risk-cue-assisted discrete A3 controller that augments the reinforcement-learning state with a history-conditioned handover-risk cue and jointly selects time-to-trigger (TTT) and handover margin (HOM) from a 63-action operator-configurable grid. The design also incorporates weak-coverage forced handover (WCFH) as a safeguard for persistently weak serving links that remain outside standard A3 activation. The framework is evaluated through an ns-3.38 simulation procedure over a 5km×5km multi-cell grid with 25 base stations and low-mobility trajectories at 1ms−1, 3ms−1 and 6ms−1. In the five-seed evaluation, relative to a fixed-threshold A3 baseline, the safeguarded cue-assisted value controller showed lower completed-event ping-pong handover and post-handover radio-link-failure rates, together with a 3.9–5.0 dB higher average post-handover RSRP gain in the 3ms−1 and 6ms−1 cases; the 1ms−1 case is retained as a low-opportunity boundary. Additional comparisons with DDQN, D3QN, and categorical PPO indicate favorable completed-event trends in the evaluated discrete-action setting, while the effect of the cue depends on the DRL backbone and event denominator. Cue-provider analysis separates the deterministic rule cue, retained as an exact audit reference for the current label definition, from the learned probabilistic CatBoost interface, and trigger-level diagnostics show that WCFH shifts control exposure toward weak-coverage intervention opportunities. These results provide bounded directional evidence of favorable completed-event behavior within a controlled and reproducible discrete A3 control framework under the studied low-mobility setting.}
}
```
