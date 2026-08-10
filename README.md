# Network-Optimization

Open-source research packages accompanying a series of publications on intelligent cellular-network optimization. The repository covers physical cell identity (PCI) planning, random-access optimization, mobility and handover control, discontinuous reception (DRX), and reinforcement-learning-based network management.

Each numbered directory is organized as a paper-associated package. Depending on the study, a package may contain source code, figure-source data, released datasets, model artifacts, simulation sources, and project-specific reproducibility notes. Please consult the `README.md` inside each project directory for the exact scope of the corresponding release.

## Repository structure

The tree below is intentionally limited to two levels.

```text
.
├── README.md
├── LICENSE
├── LICENSE-DATA
├── 01_Hybrid_PCI/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── LICENSE
│   └── LICENSE-DATA
├── 02_DQN_RACH/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── LICENSE-DATA
├── 03_DQN_PCI/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── LICENSE-DATA
├── 04_DRL_PCI/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── LICENSE-DATA
├── 05_DQN_HO/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── LICENSE-DATA
├── 06_NSA_DRX/
│   ├── src/
│   ├── figure_source_data/
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── LICENSE-DATA
└── 07_HARCADO/
    ├── src/
    ├── tools/
    ├── simulator/
    ├── ns3_dataset/
    ├── figure_source_data/
    ├── README.md
    ├── requirements.txt
    ├── MD5SUMS.txt
    ├── LICENSE
    └── LICENSE-DATA
```

## Project index

| Project | Research focus | Associated publication |
|---|---|---|
| [`01_Hybrid_PCI`](./01_Hybrid_PCI/) | Cluster-based hybrid PCI configuration and optimization for 5G EN-DC heterogeneous networks | [1] |
| [`02_DQN_RACH`](./02_DQN_RACH/) | DQN-based RACH resource optimization for coexisting H2H and M2M traffic | [2] |
| [`03_DQN_PCI`](./03_DQN_PCI/) | DQN-based PCI allocation with dynamic adjustment mechanisms | [3] |
| [`04_DRL_PCI`](./04_DRL_PCI/) | DRL-based PCI allocation with hash initialization, MOSA, FAGA, and guiding policies | [4] |
| [`05_DQN_HO`](./05_DQN_HO/) | MAR-aided multivariate DQN for adaptive UE handover management | [5] |
| [`06_NSA_DRX`](./06_NSA_DRX/) | Q-learning and Long Short Term Reward optimization for NSA-DRX | [6] |
| [`07_HARCADO`](./07_HARCADO/) | Handover-risk-cue-assisted discrete A3 control for low-mobility multi-cell networks | [7] |

## Associated publications

The references below use **IEEE citation style**, which is well aligned with the communications, networking, and engineering scope of this repository.

[1] P. Li, H. Yang, I. Kim, Z. Liu, and S. Li, “Cluster-Based Hybrid Approach for PCI Configuration and Optimization in 5G EN-DC Heterogeneous Networks,” *Journal of Network and Systems Management*, vol. 32, no. 2, Art. no. 24, 2024. DOI: [10.1007/s10922-023-09799-0](https://doi.org/10.1007/s10922-023-09799-0).

[2] X. Liu, H. Yang, S. Li, Z. Liu, and X. Lian, “Enhanced RACH Optimization in IoT Networks: A DQN Approach for Balancing H2H and M2M Communications,” *Internet of Things*, vol. 28, Art. no. 101433, 2024. DOI: [10.1016/j.iot.2024.101433](https://doi.org/10.1016/j.iot.2024.101433).

[3] J. Li, H. Yang, Z. Liu, Y. Ming, and X. Ren, “Enhanced PCI Allocation in Heterogeneous Networks: A Deep Reinforcement Learning Approach with Dynamic Adjustments,” *Computer Communications*, vol. 251, Art. no. 108494, 2026. DOI: [10.1016/j.comcom.2026.108494](https://doi.org/10.1016/j.comcom.2026.108494).

[4] J. Li, H. Yang, S. Li, Z. Liu, and W. Wang, “Adaptive PCI Allocation in Heterogeneous Networks: A DRL-Driven Framework With Hash Table, FAGA, and Guiding Policies,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 11, no. 4, pp. 2456–2472, 2025. DOI: [10.1109/TCCN.2024.3502510](https://doi.org/10.1109/TCCN.2024.3502510).

[5] W. Wang, H. Yang, S. Li, X. Liu, and Z. Wan, “Adaptive UE Handover Management with MAR-Aided Multivariate DQN in Ultra-Dense Networks,” *Journal of Network and Systems Management*, vol. 33, no. 1, Art. no. 17, 2025. DOI: [10.1007/s10922-024-09895-9](https://doi.org/10.1007/s10922-024-09895-9).

[6] X. Lian, H. Yang, S. Li, Z. Liu, X. Liu, and W. Wang, “Enhanced NSA-DRX Mechanism for Cognitive 5G Networks Utilizing Q-Learning and Long Short Term Rewards,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 12, pp. 1963–1977, 2026. DOI: [10.1109/TCCN.2025.3600997](https://doi.org/10.1109/TCCN.2025.3600997).

[7] Z. Yue, H. Yang, S. Li, J. Feng, N. A. Khan, H. Guo, Y. Kong, L. Zhang, and S. Chen, “Handover-Risk-Cue-Assisted Discrete A3 Control for Low-Mobility Multi-Cell Networks,” *Computer Networks*, vol. 288, Art. no. 112638, 2026. DOI: [10.1016/j.comnet.2026.112638](https://doi.org/10.1016/j.comnet.2026.112638).

## Reproducibility and package scope

The packages preserve the research artifacts currently released for the associated studies. Reproducibility scope is project-specific: some directories provide compact source implementations and figure-source data, while others include additional simulation sources, datasets, model artifacts, or archival implementations. Claims about exact historical experiment snapshots are therefore made only where supported by the corresponding project README.

For execution details, dependencies, figure-source coverage, and project-specific limitations, use the `README.md` in the relevant project directory as the authoritative package-level guide.

## License

This repository uses a two-level licensing structure.

- **Source code:** unless otherwise specified, source code is released under the repository-level [MIT License](./LICENSE).
- **Datasets and figure-source artifacts:** unless otherwise specified, released datasets and figure-source artifacts are provided under the repository-level [Creative Commons Attribution 4.0 International License (CC BY 4.0)](./LICENSE-DATA).
- **Project-level precedence:** a `LICENSE`, `LICENSE-DATA`, or other explicit licensing notice inside an individual project directory takes precedence for materials within that project.
- **Third-party materials:** third-party software, model artifacts, or other externally licensed materials remain subject to their respective terms when separately identified.
- **Publications:** the licenses in this repository do not alter the copyright or reuse terms of publisher-formatted articles or manuscript PDFs.

When reusing released materials, please preserve attribution and cite the associated publication.
