# Hybrid PCI

Open-source implementation accompanying the paper:

**Cluster-Based Hybrid Approach for PCI Configuration and Optimization in 5G EN-DC Heterogeneous Networks**

## Package structure

```text
src/                    Core Hybrid PCI runtime
└── rewrite_v2.py       Main implementation
figure_source_data/     Manuscript figure source data
```

## Runtime components

The `src/` directory contains the released Hybrid PCI implementation:

- EN-DC heterogeneous network abstraction;
- clustered eNB/gNB topology generation;
- PCI initialization and self-configuration;
- PCI collision and confusion detection;
- Symmetrical Comparison (SC) algorithm;
- Symmetrical Triangular Cycling (STC) algorithm;
- PCI optimization evaluation.

## Reproducibility

The package is organized to reproduce the reported simulation workflow of the manuscript using the released source code and figure-source data.

The implementation provides configurable experiments over different PCI pool sizes and eNB/gNB deployment scales.

## Integrity

Checksum information can be added for released artifacts when additional source data are included.

## License

- Source code is released under the MIT License (`LICENSE`).
- Figure-source data are released under the CC BY 4.0 License (`LICENSE-DATA`).

When reusing this package, please preserve attribution information and cite the associated manuscript.

## Cite Our Work

```bibtex
@article{li2024cluster,
  title = {Cluster-Based Hybrid Approach for {PCI} Configuration and Optimization in {5G EN-DC} Heterogeneous Networks},
  author = {Li, Pengzhao and Yang, Heng and Kim, Iksang and Liu, Zhenyu and Li, Shanshan},
  journal = {Journal of Network and Systems Management},
  volume = {32},
  number = {2},
  pages = {24},
  year = {2024},
  issn = {1064-7570},
  publisher = {Springer},
  doi = {10.1007/s10922-023-09799-0},
  url = {https://link.springer.com/article/10.1007/s10922-023-09799-0},
  keywords = {Physical Cell Identity (PCI), EN-DC architecture, heterogeneous networks, PCI collision and confusion, 5G network optimization},
  abstract = {With the development of 5G technologies and the implementation of EN-DC architecture in heterogeneous networks, managing Physical Cell Identity (PCI) has become increasingly complex. EN-DC, facilitating the coexistence of eNBs and gNBs, creates a densely populated environment that heightens the risk of PCI collisions and confusions. This study introduces a novel hybrid approach to PCI configuration in EN-DC networks, integrating centralized and distributed strategies. By organizing the network into clusters and employing newly introduced algorithms, Symmetrical Comparison (SC) and Symmetrical Triangular Cycling (STC), the method efficiently identifies and resolves PCI confusions. Simulations were conducted to evaluate the effectiveness of the proposed model under various scenarios, revealing its proficiency in preventing PCI confusion and mod 30 collisions. The results underscore the critical role of PCI pool size and offer insights into network planning and optimization. Despite some challenges in handling specific collisions, such as mod 3 and mod 4, the study suggests that incorporating reinforcement learning techniques could provide more adaptive solutions, laying the foundation for future research in this area. The research contributes to the evolving landscape of 5G EN-DC networks, emphasizing the importance of intelligent design and meticulous planning in network management.}
}
```
