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

When reusing this package, please preserve attribution information and cite the associated manuscript.

## Cite Our Work

```bibtex
% BibTeX entry will be added manually.
```
