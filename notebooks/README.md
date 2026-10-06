# Perturb-Bench: Exploratory Data Analysis & Benchmark Walkthrough Notebooks

This directory contains five interactive Jupyter notebooks designed for in-depth data exploration, biological validation, quality auditing, and baseline benchmarking prior to training machine learning models.

---

## Notebook Overview

| Notebook | Focus & Biological Context | Dataset(s) Explored | Key Concepts |
|---|---|---|---|
| [`01_dataset_overview_and_verification.ipynb`](01_dataset_overview_and_verification.ipynb) | Cross-Dataset Quality Audit & Inventory | All 6 datasets (`dixit`, `adamson`, `norman`, `replogle_*`) | Perturb-seq technologies (cut vs CRISPRi vs CRISPRa), dataset scale, sparsity ($CP10k + \log 1p$), unseen-perturbation split verification (zero leak), and loader helpers. |
| [`02_unfolded_protein_response_adamson2016.ipynb`](02_unfolded_protein_response_adamson2016.ipynb) | Cellular Stress & ER Proteostasis | `adamson_2016_ready2train.h5ad` | The Unfolded Protein Response (UPR; IRE1, PERK, ATF6 branches), on-target CRISPRi knockdown efficiency, differential expression ($\Delta$), and functional perturbation clustering. |
| [`03_combinatorial_interactions_norman2019.ipynb`](03_combinatorial_interactions_norman2019.ipynb) | Genetic Epistasis & Combinatorial Synergy | `norman_2019_ready2train.h5ad` | Combinatorial dual-gene activation with CRISPRa (dCas9-VPR), on-target overexpression, linear additivity ($\Delta A + \Delta B$) vs observed epistasis ($\Delta_{AB}$), and erythroid lineage TFs. |
| [`04_genome_wide_and_cross_cell_line_replogle2022.ipynb`](04_genome_wide_and_cross_cell_line_replogle2022.ipynb) | Scaling & Cross-Cell-Line Transfer | `replogle_k562_essential`, `replogle_rpe1`, `replogle_k562_gwps` | Core vs cell-type specific essentiality, memory-safe backed streaming, 2,055 shared perturbations between K562 (leukemia) and RPE1 (retinal epithelial), and GWPS feature selection. |
| [`05_benchmark_task_and_baselines_walkthrough.ipynb`](05_benchmark_task_and_baselines_walkthrough.ipynb) | Benchmark Task & Literature Baselines | `dixit_2016_ready2train.h5ad` | Memory-safe PyTorch `PerturbationDataset` & `DataLoader`, ground-truth test evaluation, Top-20 DE genes, Control Mean baseline, Mean Shift baseline, and benchmark metrics table. |

---

## Recommended Progression

1. **Start with `01_dataset_overview_and_verification.ipynb`**: Get a high-level view of all 6 datasets, understand the directory structure, and see the standardized loader pattern.
2. **Dive into Biology with `02_` and `03_`**:
   - `02_`: Understand single-target CRISPR interference and how to assess on-target knockdown.
   - `03_`: Understand combinatorial interactions and why predicting 2-gene perturbations requires modeling epistasis beyond linear sums.
3. **Explore Scale & Transfer with `04_`**: Learn how to handle multi-gigabyte datasets without RAM exhaustion and examine transferability between cancer and normal cells.
4. **Build Baselines with `05_`**: Complete end-to-end benchmark walkthrough implementing the literature-standard evaluation harness before training deep learning architectures.

---

## Environment & Requirements

The notebooks use the project's standard Python environment (tested with Python 3.11/3.12, PyTorch with MPS, AnnData, Scanpy, Seaborn, Matplotlib):

```bash
# If running in VSCode or JupyterLab:
jupyter lab notebooks/
```

All notebooks dynamically resolve data paths whether launched from the `notebooks/` directory or from the project root.
