# DQN RACH

Open-source implementation accompanying the paper:

**Enhanced RACH Optimization in IoT Networks: A DQN Approach for Balancing H2H and M2M Communications**

## Package structure

```text
src/                    Main DQN-based RACH implementation
└── DQN_RACH.py         Training, testing, RACH simulation, KPI, and plotting workflow
figure_source_data/     Released manuscript figure-source data
requirements.txt        Python dependencies
LICENSE                 Source-code license
LICENSE-DATA            Figure-source data license
```

## Runtime components

The released implementation includes:

- a 54-preamble random-access resource pool;
- H2H and M2M traffic, each divided into high-, medium-, and low-priority groups;
- a six-feature DQN state describing preamble allocations across the six user groups;
- a joint six-dimensional action space with decrease / no-change / increase choices for each allocation dimension;
- a fully connected DQN with a 6-feature input layer, 2000-neuron hidden layer, and 729 joint actions;
- epsilon-greedy action selection;
- User Priority-based Dynamic Adjustment (UP-DA);
- Block Rate-based Dynamic Adjustment (BR-DA);
- served, unserved, and blocked-user tracking with preamble recycling;
- access-success, blocking, delay, utilization, collision, and service-time evaluation;
- RMS-based and dual RMS-Q / RMS-loss convergence-check options.

## Implementation notes

The available project implementation follows the same core DQN + UP-DA + BR-DA RACH-management route described in the manuscript. Two implementation details differ from the final manuscript specification and are retained here as part of the archived research code:

1. **Reward calculation.** The manuscript defines the reward using separate weighted H2H and M2M blocking-rate terms together with preamble-limit and unserved-user penalties. In `DQN_RACH.py`, `weighted_reward_calculation()` additionally combines the H2H and M2M weighted blocking rates using traffic-type weights (`0.7` for H2H and `0.3` for M2M) and applies piecewise penalties according to the resulting aggregate blocking rate. The training loop subsequently adds the preamble-range and unserved-user penalties. The implemented reward is therefore not algebraically identical to the manuscript reward equations, although both optimize the same blocking- and resource-allocation objectives.

2. **Preamble holding / service duration.** The manuscript describes H2H preamble holding durations of 2--8 frames and M2M durations of 1--3 frames. In the available implementation, the active H2H service-duration logic is compressed mainly to one or two frames, while active M2M branches use a one-frame service duration; older randomized alternatives remain in comments. This changes the detailed resource-occupancy timing but not the six-group DQN allocation structure or the UP-DA / BR-DA control route.

These differences are documented rather than modified so that the released source remains identical to the available research-project implementation.

## Figure source data

`figure_source_data/` contains the currently released source data for the manuscript's main evaluation figures, covering Figs. 11--29, including the multi-panel data for Figs. 18 and 28.

The cross-scheme comparison source data for Figs. 30--32 are not included in the current release.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

The main runtime is interactive and can be started with:

```bash
python src/DQN_RACH.py
```

Runtime prompts select convergence-check, loop, traffic-generation, and blocking-rate evaluation settings.

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated manuscript.

## Cite Our Work

BibTeX entry to be added manually.
