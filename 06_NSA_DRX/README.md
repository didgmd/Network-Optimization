# NSA-DRX with Q-Learning and LSTR

Open-source implementation accompanying the paper:

**Enhanced NSA-DRX Mechanism for Cognitive 5G Networks Utilizing Q-Learning and Long Short Term Rewards**

## Package structure

```text
src/                    NSA-DRX simulation and tabular Q-learning runtime
├── QL_DRX.py           Main Q-learning and LSTR entry point
├── Parameters.py       DRX, traffic, transition, and runtime parameters
├── Classes.py          Eight-state NSA-DRX state definitions
├── StateChanger.py     Semi-Markov transitions, delay/power metrics, and rewards
├── ActionChooser.py    Epsilon-greedy action selection
└── DebugPrint.py       Debug/output utilities
figure_source_data/     Released manuscript figure-source data
requirements.txt        Python dependencies
LICENSE                 Source-code license
LICENSE-DATA            Figure-source data license
```

## Runtime components

The released implementation preserves the available research code associated with the enhanced NSA-DRX framework, including:

- an eight-state semi-Markov NSA-DRX process spanning RRC_CONNECTED and RRC_IDLE operation;
- ETSI-inspired traffic-arrival modeling and probabilistic DRX state transitions;
- analytical calculation of state occupancy, power-saving behavior, and average delay;
- tabular Q-learning for adaptive short-sleep scheduling in the short-sleep state;
- epsilon-greedy action selection over allowable short-sleep-cycle indices;
- short-term reward updates for local short-sleep decisions;
- long-term reward updates across accumulated NSA-DRX transitions;
- loss tracking and convergence-oriented training control.

## Reproducibility scope

The released implementation follows the core NSA-DRX, tabular Q-learning, and Long Short Term Reward (LSTR) route described in the associated manuscript.

The package preserves the available research implementation and its original modular structure for reproducibility-oriented inspection and experimentation. Runtime parameters are defined in `src/Parameters.py`, while the Q-learning configuration is initialized in `src/QL_DRX.py`.

Python 3.11 is recommended, consistent with the simulation environment reported in the associated manuscript.

## Figure source data

`figure_source_data/` contains the currently released source data for the principal numerical evaluation figures:

```text
Fig8.csv      → final-paper Fig. 8: Q-learning/LSTR convergence and power-delay behavior
Fig10.csv     → final-paper Fig. 10: inactivity-timer and short-cycle-timer sensitivity
Fig11.csv     → final-paper Fig. 11: long-sleep-duration sensitivity
Fig12.csv     → final-paper Fig. 12: paging-cycle-duration sensitivity
Fig13a.csv    → final-paper Fig. 13(a): RRC_CONNECTED arrival-rate delay
Fig13b.csv    → final-paper Fig. 13(b): RRC_IDLE arrival-rate delay
Fig14a.csv    → final-paper Fig. 14(a): enhanced vs. general NSA-DRX power saving
```

Other manuscript figures without standalone CSV source files are not included as separate figure-source artifacts in the current release.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

The main runtime can be started with:

```bash
python src/QL_DRX.py
```

The implementation uses PyTorch tensors for the tabular Q-table and automatically selects CUDA when available; no deep neural network is used by the released Q-learning runtime.

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated manuscript.

## Cite Our Work

BibTeX entry to be added manually.
