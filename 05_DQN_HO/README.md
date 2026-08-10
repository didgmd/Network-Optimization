# Adaptive UE Handover Management with MAR-Aided Multivariate DQN in Ultra-Dense Networks

Open-source archival package accompanying the paper:

**Adaptive UE Handover Management with MAR-Aided Multivariate DQN in Ultra-Dense Networks**

> This release is an archived/reference implementation package. Due to unavailable historical execution snapshots, the released runtime should not be interpreted as the exact code snapshot used to generate every reported manuscript result.

## Package structure

```text
src/                    Archived runtime implementation
figure_source_data/     Manuscript figure-source data
```

## Runtime components

The `src/` directory preserves the available Python implementation from the associated research project, including:

- ultra-dense network topology generation;
- Lévy Walk user mobility model;
- three DQN-based handover modules:
  - handover decision;
  - A2/A4 threshold adjustment;
  - target base-station selection;
- Memory Anchor Repository (MAR) mechanism;
- KPI evaluation utilities.

## Reproducibility scope

The package provides the available implementation artifacts and figure-source data associated with the study.

Because the exact historical experiment snapshot is unavailable, this release is intended for archival reference and code inspection rather than a claim of exact manuscript-result reproduction.

## Figure source data

The `figure_source_data/` directory contains the source data corresponding to manuscript evaluation figures.

## Requirements

Python dependencies are listed in `requirements.txt`.

## License

- Source code: MIT License (`LICENSE`).
- Figure-source data: CC BY 4.0 (`LICENSE-DATA`).

Please preserve attribution and cite the associated manuscript when reusing this package.

## Cite Our Work

```bibtex
@article{wang2025adaptive,
  title={Adaptive ue handover management with mar-aided multivariate dqn in ultra-dense networks},
  author={Wang, Weiran and Yang, Heng and Li, Shanshan and Liu, Xue and Wan, Zhaojun},
  journal={Journal of Network and Systems Management},
  volume={33},
  number={1},
  pages={17},
  year={2025},
  publisher={Springer},
  doi={10.1007/s10922-024-09895-9},
  url={https://link.springer.com/article/10.1007/s10922-024-09895-9},
  keywords={Handover, Deep Q-network, Memory anchor repository, Mobility management, Reinforcement learning},
  abstract={Ultra-Dense Networks (UDNs) are a cornerstone of 5G, offering high-speed transmission and efficient resource management. However, managing frequent handovers in UDNs poses significant challenges, including increased handover failures and frequent triggering, which degrade user experience. This paper proposes an adaptive handover management approach using a multivariate Deep Q-Network (DQN) framework integrated with a Memory Anchor Repository (MAR) mechanism. The framework consists of three DQN models: $\boldsymbol{D}_\mathrm{Dec}$ for handover decision-making, $\boldsymbol{D}_\mathrm{TH}$ for adaptive adjustment of A2 and A4 thresholds, and $\boldsymbol{D}_\mathrm{Tar}$ for target base station selection. These models leverage real-time features such as user location, movement direction, Signal-to-Interference-plus-Noise Ratio (SINR), and Reference Signal Received Power (RSRP). The MAR systematically stores and updates handover success rates at anchor points, enabling the system to learn from historical data and dynamically optimize handover decisions. Simulations conducted in a controlled UDN environment demonstrate that the proposed framework significantly reduces unnecessary handover attempts and failures. After 1250 training iterations, the overall handover failure rate decreases from 35% to 25%, with optimal performance observed using 25 anchor points. These results illustrate the framework’s potential to enhance UDN handover processes, improve overall Quality of Service (QoS), and elevate user experience.}
}
```

