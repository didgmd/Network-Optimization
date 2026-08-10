# DQN RACH

Open-source implementation accompanying the paper:

**Enhanced RACH Optimization in IoT Networks: A DQN Approach for Balancing H2H and M2M Communications**

## Package structure

```text
src/
└── DQN_RACH.py         Main DQN-based RACH simulation and training workflow

figure_source_data/     Released manuscript figure-source data

requirements.txt        Python dependencies
LICENSE                 Source-code license
LICENSE-DATA            Figure-source data license
```

## Runtime components

The released implementation contains the main DQN-based RACH optimization workflow, including:

- a 54-preamble random-access resource pool;
- H2H and M2M traffic with three priority levels;
- six-feature state representation based on preamble allocation;
- joint multi-dimensional action selection for dynamic preamble adjustment;
- fully connected DQN policy learning;
- epsilon-greedy exploration and exploitation;
- User Priority-based Dynamic Adjustment (UP-DA);
- Block Rate-based Dynamic Adjustment (BR-DA);
- user tracking, preamble recycling, and congestion handling;
- access success, delay, blocking, utilization, collision, and service-time evaluation;
- RMS-based convergence monitoring and dual convergence-check option.

## Implementation notes

The released code follows the core DQN + UP-DA + BR-DA RACH-management route described in the associated manuscript. The package preserves the available research implementation and provides the corresponding runtime workflow, while maintaining the original experimental implementation structure.

## Figure source data

`figure_source_data/` contains the currently released source data for the main evaluation figures, covering Figs. 11--29, including multi-panel source files where applicable.

The cross-scheme comparison source data for Figs. 30--32 are not included in the current release.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

The main runtime can be started with:

```bash
python src/DQN_RACH.py
```

The script provides interactive selections for convergence checking, simulation modes, traffic generation, and evaluation settings.

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

Please preserve attribution information when reusing this package and cite the associated manuscript.

## Cite Our Work

BibTeX entry to be added manually.
