# H-MADQN SAGIN

Open-source research package accompanying the paper:

**Two-Timescale Capacity-Aware Hierarchical Multi-Agent Deep Q-Network for Joint Association and Resource–Mobility Control in Space–Air–Ground Integrated Networks**

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
figure_source_data/          Manuscript figure-source data
requirements.txt             Python dependencies
LICENSE                      Source-code license
LICENSE-DATA                 Figure-source data license
```

## Framework scope

The implementation includes:

- a heterogeneous SAGIN topology with satellite, macro-cell, small-cell, and UAV tiers;
- two-timescale hierarchical multi-agent control with slow-timescale cloud association and fast-timescale edge resource/mobility control;
- capacity-aware user association with a Sequential Decision Mechanism (SDM);
- Entry-Indexed Alignment and Padding (EIAP) for fixed-dimensional learning under changing served-user sets;
- Joint Output Decoupling Strategy (JODS) for feasible hybrid multi-user scheduling actions;
- OFDMA resource scheduling and UAV mobility control;
- cloud and edge reward design incorporating QoS, throughput, load/capacity, fairness, and association stability.

The default configuration in `src/Parameters.py` uses a 10 km × 10 km area with one satellite, two macro BSs, four small BSs, two UAV BSs, and 100 users. The default cloud association period is `K = 3`.

## Experiment runners

```text
baselines.py             → baseline comparisons
k_overhead.py            → K-sensitivity and cloud signaling overhead
channel_shadowing.py     → channel-shadowing robustness
scalability_density.py   → UE-scale and hotspot-density evaluation
sdm_reward.py            → SDM-order and reward-component sensitivity
```

## Figure source data

`figure_source_data/` contains the numerical source data for the manuscript's data-backed figures:

```text
fig3_losses.csv                           → Fig. 3: agent training losses
fig4_rewards.csv                          → Fig. 4: agent rewards
fig5_baseline_comparison.csv              → Fig. 5: baseline comparison
fig6_k_sensitivity.csv                    → Fig. 6: association-period K sensitivity / overhead
fig7_shadowing_robustness.csv             → Fig. 7: shadowing robustness
fig8_scalability_density.csv              → Fig. 8: scalability and user-density evaluation
fig9_sdm_order_sensitivity.csv            → Fig. 9: SDM ordering sensitivity
fig10_exploration_kpi_dynamics.csv        → Fig. 10: exploration and KPI dynamics
fig11_reward_component_ablation.csv       → Fig. 11: reward-component ablation
fig12_stability_fairness_tradeoff.csv     → Fig. 12: stability–fairness trade-off
fig13_kpi_correlation_matrices.csv        → Fig. 13: KPI correlation matrices
fig14_kpi_distribution.csv                → Fig. 14: KPI distributions
fig15_satellite_availability_ablation.csv → Fig. 15: satellite-availability ablation
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

The main runtime can be started with:

```bash
python src/SC_RL_main.py
```

Individual experiment runners expose their own command-line options, for example:

```bash
python src/baselines.py --help
```

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated paper.

## Cite Our Work

<!-- BibTeX citation will be added after publication. -->
