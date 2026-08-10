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

```bibtex
@article{LIU2024101433,
  title = {Enhanced {RACH} Optimization in {IoT} Networks: A {DQN} Approach for Balancing {H2H} and {M2M} Communications},
  author = {Liu, Xue and Yang, Heng and Li, Shanshan and Liu, Zhenyu and Lian, Xiaohui},
  journal = {Internet of Things},
  volume = {28},
  pages = {101433},
  year = {2024},
  issn = {2542-6605},
  publisher = {Elsevier},
  doi = {10.1016/j.iot.2024.101433},
  url = {https://www.sciencedirect.com/science/article/pii/S2542660524003743},
  keywords = {Cellular-based IoT, Deep Q-network, Machine-to-machine communications, Random access, User priority},
  abstract = {A novel adaptive Deep Q-Network (DQN)-based algorithm is designed for the dynamic management of the Random Access Channel (RACH) in LTE networks, facilitating the coexistence of Human-to-Human (H2H) and Machine-to-Machine (M2M) communications. This algorithm employs the integration of user priority and block rate-based dynamic adjustment policies within the DQN framework, significantly enhancing service quality across cellular communications. By categorizing devices into three priority tiers based on their Quality of Service (QoS) requirements, the scheme enables dynamic allocation of RACH resources, thus effectively reducing collisions and enhancing network efficiency. Additionally, the implementation of a dual-criteria convergence check within the model ensures the algorithm’s robustness and reliability, offering a significant advancement in managing the intricate dynamics of M2M and H2H communications. This approach not only exhibits effectiveness in access success rates, reductions in access delay, and increased preamble utilization but also underscores the potential for further refinements in learning efficiency and overall performance through dynamic parameter adjustments. This innovative study offers valuable insights into optimizing RACH resources and sets a solid foundation for advancing intelligent network management in increasingly complex communication landscapes.}
}
```

