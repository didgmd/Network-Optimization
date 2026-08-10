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
  title={Cluster-based hybrid approach for PCI configuration and optimization in 5G EN-DC heterogeneous networks},
  author={Li, Pengzhao and Yang, Heng and Kim, Iksang and Liu, Zhenyu and Li, Shanshan},
  journal={Journal of Network and Systems Management},
  volume={32},
  number={2},
  pages={24},
  year={2024},
  publisher={Springer},
  doi = {10.1007/s10922-023-09799-0},
  url = {https://link.springer.com/article/10.1007/s10922-023-09799-0}
}
```
