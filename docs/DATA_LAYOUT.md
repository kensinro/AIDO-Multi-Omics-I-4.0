# DATA_LAYOUT

Default local folders expected by the scripts:

```text
D:/AIDO-Data/
├── UCSC_XENA/
│   ├── Breast Cancer (BRCA)/
│   │   ├── GE.tsv
│   │   ├── CN.tsv
│   │   ├── MU.tsv or MU_fixed.tsv
│   │   ├── Phenotype.tsv
│   │   ├── TCGA.BRCA.sampleMap_BRCA_clinicalMatrix
│   │   └── optional BRCA_stage_groups_from_survival.tsv
│   └── ... other TCGA cancer folders ...
├── GSEA/
│   ├── h.all.v2026.1.Hs.symbols.gmt
│   ├── optional c5.go.bp.v2026.1.Hs.symbols.gmt
│   └── optional c2.cp.reactome.v2026.1.Hs.symbols.gmt
└── External/
    ├── brca_metabric/
    ├── ucec_cptac_2020/
    ├── luad_cptac_2020/
    ├── rcc_cptac_gdc/
    └── kirp_tcga_pan_can_atlas_2018/
```

Edit the CONFIG blocks in the scripts if your local data paths differ.

The repository intentionally excludes raw data files and generated output folders.
