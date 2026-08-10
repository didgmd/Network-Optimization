# DQN PCI

Open-source implementation accompanying the paper:

**Enhanced PCI Allocation in Heterogeneous Networks: A Deep Reinforcement Learning Approach with Dynamic Adjustments**

## Package structure

```text
src/                    DQN-based PCI allocation runtime
├── PCI_DQN_main.py     Main training and testing entry point
├── QNet.py             Q-network definition
├── NetworkTopology.py  Heterogeneous-network topology construction
├── PciAllocation.py    PCI pool and initial allocation
├── NeighborList.py     Neighbor-relation construction
├── ActionChooser.py    DQN action selection
├── StateChanger.py     PCI reassignment and state transition
├── RewardCalculator.py PCI conflict/confusion evaluation and reward
├── Normalization.py    State normalization
├── Validation.py       Testing and validation workflow
├── FigurePlotter.py    Runtime visualization utilities
├── DataSave.py         Runtime result export
└── DebugPrint.py       Debug/output utilities
figure_source_data/     Released manuscript figure-source data
requirements.txt        Python dependencies
LICENSE                 Source-code license
LICENSE-DATA            Figure-source data license
```

## Runtime components

The released implementation preserves the available research code for the DQN-based centralized PCI-allocation framework, including:

- heterogeneous eNB/gNB topology generation and neighbor construction;
- PCI pool management and PCI reassignment;
- PCI conflict/confusion detection and penalty-based reward calculation;
- DQN-based node selection with EMA-based value processing;
- guided target-value calculation;
- R-DAM, E-DAM, and A-DAM dynamic adjustment mechanisms;
- training, testing, visualization, and result-export utilities.

## Reproducibility scope

The released implementation follows the core DQN-based centralized PCI-allocation route described in the associated manuscript, including EMA-based value smoothing, guided target calculation, and dynamic adjustment mechanisms.

The package preserves the available research implementation and its modular structure for reproducibility-oriented inspection and experimentation.

## Figure source data

`figure_source_data/` contains the currently released source data for the principal evaluation figures. The filenames follow the final manuscript numbering:

```text
Fig9.csv                         → final-paper Fig. 9 (EMA loss)
Fig11.csv                        → final-paper Fig. 11
Fig12.csv                        → final-paper Fig. 12
Fig13_Base.csv + Fig13_Plot.csv  → final-paper Fig. 13
Fig14a.csv–Fig14c.csv            → final-paper Fig. 14(a–c)
Fig15a.csv–Fig15c.csv            → final-paper Fig. 15(a–c)
```

Runtime-generated visualizations without standalone CSV source files are not included as separate figure-source artifacts. Aggregate source data for final-paper Fig. 16 are not included in the current release.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

The main runtime can be started with:

```bash
python src/PCI_DQN_main.py
```

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated manuscript.

## Cite Our Work

```bibtex
@article{LI2026108494,
	title = {Enhanced PCI allocation in heterogeneous networks: A deep reinforcement learning approach with dynamic adjustments},
	journal = {Computer Communications},
	volume = {251},
	pages = {108494},
	year = {2026},
	doi = {10.1016/j.comcom.2026.108494},
	author = {Jiani Li and Heng Yang and Zhenyu Liu and Yibo Ming and Xia Ren}
}
```
