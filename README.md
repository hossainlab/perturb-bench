# Single-Cell Perturbation Response Prediction Benchmark

Six fully preprocessed Perturb-seq datasets for training and evaluating perturbation-response
models (e.g., predicting post-perturbation cellular expression of unseen genetic knockouts/knockdowns).
Sources are harmonized via [scPerturb](https://doi.org/10.1038/s41592-023-02144-y)
(Peidli et al., *Nature Methods* 2024); preprocessing and splits were generated
from the scPerturb `.h5ad` files.

---

## Project Structure

```text
perturb-bench/
├── configs/               # YAML configurations for models and experiments
│   ├── baselines.yaml
│   └── train_mlp.yaml
├── data/                  # Packaged *_ready2train.h5ad datasets (~32.5 GB)
├── docs/                  # Benchmark protocol & hardware guidelines
│   └── PROTOCOL.md
├── notebooks/             # Exploratory analysis & biological validation
│   ├── 01_dataset_overview_and_verification.ipynb
│   ├── 02_unfolded_protein_response_adamson2016.ipynb
│   ├── 03_combinatorial_interactions_norman2019.ipynb
│   ├── 04_genome_wide_and_cross_cell_line_replogle2022.ipynb
│   ├── 05_benchmark_task_and_baselines_walkthrough.ipynb
│   └── README.md
├── results/               # Checkpoints, evaluation logs, and benchmark tables
├── scripts/               # Reproducible CLI runners
│   ├── run_baselines.py   # Benchmark standard literature baselines
│   └── train_model.py     # Train neural perturbation response models
├── src/perturb_bench/     # Core Python package
│   ├── data/              # Memory-safe PyTorch datasets & clean loaders
│   ├── models/            # Baseline models & neural architectures
│   ├── evaluation/        # Benchmark harness & metrics (Pearson ρ, MSE)
│   └── utils/             # Hardware device detection & reproducibility seeds
├── tests/                 # Unit test suite (pytest)
├── pyproject.toml         # uv / pip dependency specification
└── uv.lock                # Locked reproducible dependency graph
```

---

## Quickstart

### 1. Environment Setup
The project uses `uv` for reproducible environment management:

```bash
# Sync environment from uv.lock
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Verify hardware acceleration:
```bash
python -c "import torch; print('Apple Silicon MPS available:', torch.backends.mps.is_available())"
```

### 2. Run Test Suite
Verify that data loaders, baseline models, neural models, and metric functions pass all unit tests:
```bash
uv run pytest tests/
```

### 3. Evaluate Literature Baselines
Evaluate the non-negotiable baselines (*Control Mean* and *Mean Shift*) on held-out test perturbations:

```bash
# On a specific dataset (e.g. Dixit, Adamson, Norman)
uv run python scripts/run_baselines.py --dataset dixit

# Across all available datasets
uv run python scripts/run_baselines.py --dataset all
```
Metrics will be saved to `results/baselines_<dataset>.json` and CSV format.

### 4. Train a Neural Perturbation Model
Train the `PerturbationResidualMLP` on Apple Silicon (MPS) or CUDA:

```bash
uv run python scripts/train_model.py --dataset dixit --epochs 10 --batch-size 128
```
Trained checkpoints and evaluation reports will be saved in `results/checkpoints/<dataset>/`.

---

## Python API Usage

```python
from perturb_bench.data import load_dataset, create_dataloaders
from perturb_bench.models import ControlMeanPredictor, MeanShiftPredictor, PerturbationResidualMLP
from perturb_bench.evaluation import BenchmarkHarness

# 1. Load dataset with automated control & barcode audit fixes
adata = load_dataset("dixit")

# 2. Fit and evaluate standard baselines
harness = BenchmarkHarness(adata)
baseline = MeanShiftPredictor().fit(adata)
summary, per_pert_df = harness.evaluate_model(baseline, model_name="Mean Shift")

print(summary)
# {'Model': 'Mean Shift', 'Pearson ρ (All)': 0.99955, 'MSE (All)': 0.00011, 'Pearson ρ (Top-20 DE)': 0.99529, ...}

# 3. Build memory-safe PyTorch DataLoaders (keeps sparse CSR in host RAM)
train_loader, val_loader, test_loader, pert2idx, idx2pert = create_dataloaders(adata, batch_size=128)
```

---

## Benchmark Datasets

| File | Cells | Genes | Perturbations | Controls | Paper |
|---|---|---|---|---|---|
| `dixit_2016_ready2train.h5ad` | 19,268 | 21,713 | 27 | 3,491 | [Dixit et al., Cell 167:1853 (2016)](https://doi.org/10.1016/j.cell.2016.11.038) |
| `adamson_2016_ready2train.h5ad` | 65,337 | 32,738 | 115 | 7,394 | [Adamson et al., Cell 167:1867 (2016)](https://doi.org/10.1016/j.cell.2016.11.048) |
| `norman_2019_ready2train.h5ad` | 111,445 | 33,694 | 236 (combinatorial) | 11,855 | [Norman et al., Science 365:786 (2019)](https://doi.org/10.1126/science.aax4438) |
| `replogle_k562_essential_ready2train.h5ad` | 310,385 | 8,563 | 2,057 | 10,691 | [Replogle et al., Cell 185:2559 (2022)](https://www.cell.com/fulltext/S0092-8674(22)00597-9) |
| `replogle_rpe1_ready2train.h5ad` | 247,914 | 8,749 | 2,393 | 11,485 | Replogle et al. 2022 (same) |
| `replogle_k562_gwps_ready2train.h5ad` | 500,000 | 7,864 | 9,847 | 75,328 | Replogle et al. 2022 (same) |

---

## Split Methodology (Unseen Perturbations)

Each **whole perturbation** is assigned to exactly one of train/val/test (80/10/10 over unique perturbations, seed 0). A perturbation never appears in more than one split, so the test task is always: **"predict the single-cell response to a genetic perturbation never seen in training."**

Non-targeting control cells are strictly assigned to `train`. For Norman combinatorial screens, the unit is the perturbation pair.

---

## Citations

- Peidli et al. "scPerturb: harmonized single-cell perturbation data." *Nature Methods* 21:99–110 (2024). https://doi.org/10.1038/s41592-023-02144-y
- Ahlmann-Eltze et al. "Systematic benchmark of models for single-cell perturbation response prediction." *Nature Methods* (2025). https://doi.org/10.1038/s41592-024-02500-1
- Original papers per dataset (table above).
