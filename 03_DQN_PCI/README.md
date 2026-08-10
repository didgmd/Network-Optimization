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
  title = {Enhanced {PCI} Allocation in Heterogeneous Networks: A Deep Reinforcement Learning Approach with Dynamic Adjustments},
  author = {Li, Jiani and Yang, Heng and Liu, Zhenyu and Ming, Yibo and Ren, Xia},
  journal = {Computer Communications},
  volume = {251},
  pages = {108494},
  year = {2026},
  issn = {0140-3664},
  publisher = {Elsevier},
  doi = {10.1016/j.comcom.2026.108494},
  url = {https://www.sciencedirect.com/science/article/pii/S0140366426000848},
  keywords = {Deep Q-network, Heterogeneous networks, PCI conflict and confusion, Physical cell identity, Reinforcement learning},
  abstract = {As 5G networks proliferate, managing network complexity and resolving PCI conflicts become increasingly challenging. This study introduces a DQN-based approach optimized for heterogeneous networks, effectively addressing PCI conflicts and confusions. Integrating reinforcement learning with neural networks, the model incorporates dynamic adjustment mechanisms—R-DAM, E-DAM, and A-DAM—to enhance adaptability and efficacy. Additionally, the guiding policy and EMA algorithm dynamically adjust expected rewards to swiftly reflect state changes, prioritizing nodes requiring PCI adjustments to expedite convergence. Simulation results confirm the model’s effectiveness in significantly reducing PCI conflicts and confusions, with loss values ($\bar{E}_\mathrm{EMA}$) starting consistently below 0.5 and reducing to magnitude of $10^{−4}$, further demonstrating robustness and adaptability. Despite these advances, challenges remain in scaling the solution to larger and more complex networks. This paper not only sets a precedent for future technological advancements in network management but also highlights the continuous need for innovation to keep pace with the evolving landscape of global communication networks.}
}
```

