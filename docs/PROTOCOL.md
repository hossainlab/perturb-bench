# Training Protocol — Perturbation Response Prediction on Apple Silicon (M4 Pro, 24 GB)

Companion to `README.md`. Assumes the six `*_ready2train.h5ad` files are
downloaded to a local folder (total ~32 GB; keep ~70 GB free on disk).

## 1. Is 24 GB enough? — Yes, with one rule

**The rule: never load a dataset as a dense matrix, and never put the whole
training matrix on the GPU.** All six files store sparse CSR matrices; keep
them sparse in RAM and densify only per-minibatch.

Realistic memory footprint (sparse matrix in RAM + model + batch):

| Dataset | File size | Sparse matrix in RAM | Fits 24 GB? |
|---|---|---|---|
| dixit_2016 | 0.7 GB | ~1 GB | trivially |
| adamson_2016 | 1.9 GB | ~2 GB | trivially |
| norman_2019 | 2.9 GB | ~3 GB | yes |
| replogle_rpe1 | 7.0 GB | ~7 GB | yes |
| replogle_k562_essential | 9.0 GB | ~9 GB | yes, alone |
| replogle_k562_gwps | 12.5 GB | ~13 GB | yes, alone — close other apps |

The M4 Pro's unified memory is shared by CPU and GPU (PyTorch "MPS" backend),
so the numbers above cover both. Expect the GPU to accelerate transformer/MLP
models well; some ops (rare scatter/spline ops) silently fall back to CPU —
check for warnings once, early.

## 2. Environment setup (one time)

```bash
# Python 3.11 via miniforge or uv (arm64 native)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch anndata scanpy scipy numpy pandas scikit-learn
python -c "import torch; print(torch.backends.mps.is_available())"  # must print True
```

Then run `python verify_package.py` in the data folder — all six files must
PASS before you write any training code.

## 3. Data loading pattern (memory-safe)

```python
import anndata as ad, torch
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp
import numpy as np

class PerturbationDataset(Dataset):
    def __init__(self, adata, split):
        mask = adata.obs["split"] == split
        self.adata = adata[mask]              # boolean indexing keeps CSR sparse
        self.perts = self.adata.obs["perturbation"].values
        self.ctrl_mean = self._control_mean(adata)

    def _control_mean(self, adata):
        ctrl = adata[adata.obs["is_control"]]
        X = ctrl.X
        return np.asarray(X.mean(axis=0)).ravel()   # baseline prediction

    def __getitem__(self, i):
        x = self.adata.X[i]
        x = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
        return torch.from_numpy(x.astype(np.float32)), self.perts[i]

# batch_size 64-256 on 24 GB; densify happens per-item, not per-dataset
loader = DataLoader(PerturbationDataset(adata, "train"), batch_size=128, shuffle=True)
```

Do NOT do `adata.X.toarray()` or `torch.tensor(adata.X)` on any file.

## 4. Task definition (what the model learns)

- **Input**: perturbation identity (gene name / gene pair) — embed it
  (e.g., via gene-expression embedding or learned embedding lookup).
- **Output**: predicted post-perturbation log-normalized expression vector
  (the full gene dimension, or the top-20 differential genes per perturbation
  — the GEARS convention).
- **Evaluation**: hold-out perturbations in `split == "test"`. Report
  per-perturbation Pearson correlation and MSE between predicted and true
  post-perturbation expression, averaged over perturbations, plus the same
  restricted to the top-20 DE genes per perturbation (literature-standard
  metrics; see Zhao et al. 2023 and Ahlmann-Eltze et al. 2025).

## 5. Baselines first (non-negotiable)

Published benchmarks show deep models often fail to beat these — implement
them before any neural net and put them in every results table:

1. **Control mean**: predict the mean control-cell profile for every
   perturbation (one line of code, surprisingly strong).
2. **Mean shift**: control mean + perturbation's average expression delta
   estimated from training perturbations (linear baseline that beat deep
   models in Ahlmann-Eltze et al., Nature Methods 2025).
3. **Ridge regression** on perturbation one-hot or gene-embedding features.

## 6. Suggested progression

1. **dixit_2016** (19k cells) — end-to-end pipeline debugging in minutes.
2. **adamson_2016 / norman_2019** — the standard benchmarks; compare against
   published GEARS/scGen numbers directly.
3. **replogle_k562_essential** (310k cells) — scale-up; watch RAM.
4. **replogle_k562_gwps** (500k cells) — only with the loading pattern above;
   close other apps; if MPS OOMs, reduce batch size or train on CPU (M4 Pro
   CPU is fast for small models).
5. **replogle_rpe1** — use as an external generalization test: train on K562,
   test on RPE1 perturbations (cross-cell-line transfer, a known open problem).

## 7. Pitfalls on Apple Silicon

- MPS OOM errors are less informative than CUDA's — if you hit one, halve the
  batch size first.
- `torch.compile` support for MPS is incomplete; skip it.
- Set `torch.manual_seed(0)` and `np.random.seed(0)`; the splits are already
  seeded, so results are reproducible across machines.
- Do not move the sparse matrix to the GPU; move dense minibatches only.
