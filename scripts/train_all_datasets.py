"""Train and benchmark BioTransfer across all Perturb-seq datasets."""
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from perturb_bench.data.loader import load_dataset, get_control_mean, compute_group_means
from perturb_bench.evaluation.benchmark import BenchmarkHarness
from perturb_bench.models.baselines import ControlMeanPredictor, MeanShiftPredictor
from perturb_bench.models.biotransfer import BioTransferNet
from perturb_bench.models.embeddings import clean_perturbation_name
from perturb_bench.utils.device import get_device
from perturb_bench.utils.seed import set_seed


class BioTransferPredictor:
    """Benchmark wrapper providing .predict(pert_name) for BenchmarkHarness."""

    def __init__(self, model: BioTransferNet, esm_dict: Dict[str, np.ndarray], control_mean: np.ndarray, gene_names: List[str], device: torch.device):
        self.model = model.eval()
        self.esm_dict = esm_dict
        self.control_mean = control_mean.astype(np.float32)
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.device = device

    def predict(self, pert_name: str) -> np.ndarray:
        genes = clean_perturbation_name(pert_name)
        vecs = [self.esm_dict[g] for g in genes if g in self.esm_dict]
        if vecs:
            esm_vec = np.mean(vecs, axis=0).astype(np.float32)
        else:
            esm_vec = np.zeros(320, dtype=np.float32)

        basal = 0.0
        if len(genes) == 1 and genes[0] in self.gene_to_idx:
            basal = float(self.control_mean[self.gene_to_idx[genes[0]]])

        with torch.no_grad():
            b_esm = torch.from_numpy(esm_vec).unsqueeze(0).to(self.device)
            b_ctrl = torch.from_numpy(self.control_mean).unsqueeze(0).to(self.device)
            b_basal = torch.tensor([[basal]], dtype=torch.float32).to(self.device)
            pred_expr, _ = self.model(b_esm, b_ctrl, b_basal)
            return pred_expr.squeeze(0).cpu().numpy()


def train_and_eval_dataset(dataset_name: str, esm_dict: Dict[str, np.ndarray], device: torch.device, epochs: int = 35) -> List[Dict[str, Any]]:
    print(f"\n========================================================")
    print(f"Training BioTransfer on: {dataset_name}")
    print(f"========================================================")

    adata = load_dataset(dataset_name)
    n_genes = adata.n_vars
    ctrl_mean = get_control_mean(adata)
    harness = BenchmarkHarness(adata)

    # 1. Fit standard baselines
    ctrl_baseline = ControlMeanPredictor().fit(adata)
    shift_baseline = MeanShiftPredictor().fit(adata)

    # 2. Extract train perturbations and empirical means
    ctrl_mask = adata.obs["is_control"].astype(bool).values
    train_mask = (adata.obs["split"] == "train").values & (~ctrl_mask)
    train_X = adata.X[train_mask]
    train_perts = adata.obs.loc[train_mask, "perturbation"].values

    unique_train_perts, train_means = compute_group_means(train_X, train_perts)

    # Build training tensors
    esm_list = []
    basal_list = []
    tgt_list = []
    gene_map = {g: i for i, g in enumerate(adata.var_names)}

    for p, mean_vec in zip(unique_train_perts, train_means):
        genes = clean_perturbation_name(p)
        vecs = [esm_dict[g] for g in genes if g in esm_dict]
        if vecs:
            e_vec = np.mean(vecs, axis=0)
        else:
            e_vec = np.zeros(320, dtype=np.float32)

        basal = 0.0
        if len(genes) == 1 and genes[0] in gene_map:
            basal = float(ctrl_mean[gene_map[genes[0]]])

        esm_list.append(e_vec)
        basal_list.append([basal])
        tgt_list.append(mean_vec)

    t_esm = torch.from_numpy(np.array(esm_list, dtype=np.float32)).to(device)
    t_ctrl = torch.from_numpy(np.tile(ctrl_mean, (len(esm_list), 1)).astype(np.float32)).to(device)
    t_basal = torch.from_numpy(np.array(basal_list, dtype=np.float32)).to(device)
    t_tgt = torch.from_numpy(np.array(tgt_list, dtype=np.float32)).to(device)

    # Train BioTransferNet
    model = BioTransferNet(n_genes=n_genes, esm_dim=320, hidden_dims=[256, 256]).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()

    ds = TensorDataset(t_esm, t_ctrl, t_basal, t_tgt)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for b_e, b_c, b_b, b_y in loader:
            opt.zero_grad()
            pred, _ = model(b_e, b_c, b_b)
            loss = crit(pred, b_y)
            loss.backward()
            opt.step()

    predictor = BioTransferPredictor(model, esm_dict, ctrl_mean, adata.var_names, device)

    # Evaluate all models on hold-out test set
    models = {
        "Control Mean": ctrl_baseline,
        "Mean Shift": shift_baseline,
        "BioTransferNet (Ours)": predictor,
    }

    results = []
    for name, m in models.items():
        summary, _ = harness.evaluate_model(m, model_name=name)
        summary["Dataset"] = dataset_name
        summary["Model"] = name
        results.append(summary)

    res_df = pd.DataFrame(results)
    print("\nResults on Hold-Out Test Perturbations:")
    cols = ["Dataset", "Model", "Pearson ρ (Top-20 DE)", "MSE (Top-20 DE)", "Pearson ρ (All)", "MSE (All)", "N Test Targets"]
    print(res_df[cols].to_string(index=False))
    return results


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    master_cache = torch.load("data/embeddings/esm2_8m_master.pt", map_location="cpu", weights_only=False)
    print(f"Loaded master ESM-2 cache ({len(master_cache):,} embeddings)")

    all_results = []
    for d in ["dixit", "adamson", "norman"]:
        res = train_and_eval_dataset(d, master_cache, device, epochs=40)
        all_results.extend(res)

    out_df = pd.DataFrame(all_results)
    cols = ["Dataset", "Model", "Pearson ρ (Top-20 DE)", "MSE (Top-20 DE)", "Pearson ρ (All)", "MSE (All)", "N Test Targets"]
    out_df = out_df[cols]

    out_path = Path("results/all_individual_datasets_summary.csv")
    out_df.to_csv(out_path, index=False)
    print("\n================ FINAL MULTI-DATASET SUMMARY ================")
    print(out_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
