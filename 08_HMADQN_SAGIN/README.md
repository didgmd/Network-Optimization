# H-MADQN SAGIN

Open-source research package accompanying the accepted paper:

**Two-Timescale Capacity-Aware Hierarchical Multi-Agent Deep Q-Network for Joint Association and Resource–Mobility Control in Space–Air–Ground Integrated Networks**

The manuscript has been accepted for publication in *Computer Networks*. Final bibliographic metadata and DOI are pending.

## Package structure

```text
src/                         H-MADQN runtime and experiment runners
├── SC_RL_main.py            Main two-timescale H-MADQN runtime
├── Parameters.py            SAGIN, learning, and reward parameters
├── Classes.py               Network entities and learning-agent classes
├── Topology.py              SAGIN topology and user construction
├── ActionChooser.py         Cloud/edge action-selection utilities
├── RewardCalculator.py      Cloud and edge reward/KPI utilities
├── DebugPrint.py            Debug/output utilities
├── baselines.py             Baseline-comparison experiments
├── k_overhead.py            Association-period and overhead experiments
├── channel_shadowing.py     Shadowing-robustness experiments
├── scalability_density.py   UE-scale and density experiments
└── sdm_reward.py            SDM-order and reward-sensitivity experiments
figure_source_data/          Released manuscript figure-source data
requirements.txt             Tested Python dependencies
LICENSE                      Source-code license
LICENSE-DATA                 Figure-source data license
```

## Framework scope

The released implementation preserves the accepted study's two-timescale hierarchical multi-agent deep Q-learning route for space–air–ground integrated networks (SAGINs). The framework includes:

- a heterogeneous SAGIN topology with satellite, macro-cell, small-cell, and UAV tiers;
- centralized training with decentralized execution (CTDE)-oriented hierarchical control;
- slow-timescale cloud association and fast-timescale edge resource/mobility control;
- capacity-aware user association with a Sequential Decision Mechanism (SDM);
- fixed-dimensional learning support for changing served-user sets through Entry-Indexed Alignment and Padding (EIAP);
- Joint Output Decoupling Strategy (JODS) for feasible hybrid multi-user scheduling actions;
- OFDMA resource scheduling across heterogeneous access tiers;
- UAV mobility control;
- cloud and edge reward scalarization incorporating QoS, throughput, load/capacity, fairness, and association-stability terms;
- epsilon-greedy learning, experience replay, target-network updates, and KPI/convergence tracking.

The default configuration in `src/Parameters.py` uses a 10 km × 10 km area with one satellite, two macro BSs, four small BSs, two UAV BSs, and 100 users. The default cloud association period is `K = 3`.

## Experiment runners

The five public experiment runners correspond to the principal robustness, sensitivity, and ablation studies developed during the accepted-paper evaluation:

```text
baselines.py             → baseline comparisons
k_overhead.py            → K-sensitivity and cloud signaling overhead
channel_shadowing.py     → channel-shadowing robustness
scalability_density.py   → UE-scale and hotspot-density evaluation
sdm_reward.py            → SDM-order and reward-component sensitivity
```

The historical research source used `revision_group1_*` through `revision_group5_*` filenames. These public runner names were simplified for release clarity. Non-executable revision-context module docstrings were removed from the corresponding release copies, and `scalability_density.py` contains the necessary import-path update from `revision_group1_baselines` to `baselines`. No scientific execution logic was changed by these release-only naming/description adjustments.

Some experiment runners retain historical `revision_round1` labels in their generated output-directory structure. These labels are implementation provenance and do not denote a separate public package or data release.

## Figure source data

`figure_source_data/` contains the released numerical source data for the manuscript's data-backed figures:

```text
fig3_losses.csv                         → Fig. 3: agent training losses
fig4_rewards.csv                        → Fig. 4: agent rewards
fig5_baseline_comparison.csv            → Fig. 5: baseline comparison
fig6_k_sensitivity.csv                  → Fig. 6: association-period K sensitivity / overhead
fig7_shadowing_robustness.csv           → Fig. 7: shadowing robustness
fig8_scalability_density.csv            → Fig. 8: scalability and user-density evaluation
fig9_sdm_order_sensitivity.csv          → Fig. 9: SDM ordering sensitivity
fig10_exploration_kpi_dynamics.csv      → Fig. 10: exploration and KPI dynamics
fig11_reward_component_ablation.csv     → Fig. 11: reward-component ablation
fig12_stability_fairness_tradeoff.csv   → Fig. 12: stability–fairness trade-off
fig13_kpi_correlation_matrices.csv      → Fig. 13: KPI correlation matrices
fig14_kpi_distribution.csv              → Fig. 14: KPI distributions
fig15_satellite_availability_ablation.csv → Fig. 15: satellite-availability ablation
```

The graphical abstract and the architecture/framework illustrations (Figs. 1–2) do not have standalone numerical CSV source files and are not included in `figure_source_data/`.

## Requirements

The tested environment is recorded in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

The main runtime can be started with:

```bash
python src/SC_RL_main.py
```

`SC_RL_main.py` provides interactive modes for full exploration, the normal H-MADQN route, reward ablations, and figure generation. Individual experiment runners expose their own command-line options; use, for example:

```bash
python src/baselines.py --help
```

## Reproducibility scope

This package is a reproducibility-oriented release of the accepted research implementation and its authoritative figure-source CSV files. The underlying accepted-paper development repository also contains internal review, validation, execution-governance, and historical log artifacts; those process materials are intentionally outside the scope of this public package.

The source package therefore preserves the scientific runtime and manuscript-facing experiment runners without presenting internal revision-governance records as public execution interfaces.

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated paper.

## Cite Our Work

Final BibTeX metadata and DOI will be added after the publisher assigns the final bibliographic record.
