"""Comprehensive Benchmark Runner for Cross-Cell Transfer & Multimodal Perturbation Prediction.

Evaluates:
1. Within-Cell Out-of-Distribution (OOD) unseen perturbation prediction (K562 Test).
2. Zero-Shot Cross-Cell Transfer on shared perturbations (K562 -> RPE1).
3. Dual Zero-Shot: Cross-Cell Transfer on unseen gene knockouts (RPE1 Test).
4. Cross-Cell Screen Translation (translating K562 CRISPR screen to RPE1).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from perturb_bench.data.transfer import CrossCellAlignedData
from perturb_bench.evaluation.metrics import extract_top20_de_indices
from perturb_bench.models.biotransfer import (
    BioTransferNet,
    BioTransferTranslator,
    ESMResidualMLP,
    IntegerResidualMLP,
)
from perturb_bench.utils.device import get_device
from perturb_bench.utils.seed import set_seed


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_ctrl: np.ndarray,
) -> Dict[str, float]:
    """Compute literature-standard Pearson rho and MSE (global and Top-20 DE)."""
    # Global metrics
    mse_all = float(np.mean((y_true - y_pred) ** 2))
    diff_true = y_true - np.mean(y_true)
    diff_pred = y_pred - np.mean(y_pred)
    denom = np.sqrt(np.sum(diff_true**2) * np.sum(diff_pred**2))
    rho_all = float(np.sum(diff_true * diff_pred) / denom) if denom > 1e-12 else 0.0

    # Top-20 DE metrics
    top20_idx = extract_top20_de_indices(y_true, y_ctrl)
    y_true_de = y_true[top20_idx]
    y_pred_de = y_pred[top20_idx]

    mse_de = float(np.mean((y_true_de - y_pred_de) ** 2))
    diff_true_de = y_true_de - np.mean(y_true_de)
    diff_pred_de = y_pred_de - np.mean(y_pred_de)
    denom_de = np.sqrt(np.sum(diff_true_de**2) * np.sum(diff_pred_de**2))
    rho_de = float(np.sum(diff_true_de * diff_pred_de) / denom_de) if denom_de > 1e-12 else 0.0

    return {
        "pearson_all": rho_all,
        "mse_all": mse_all,
        "pearson_top20_de": rho_de,
        "mse_top20_de": mse_de,
    }


def evaluate_predictions_batch(
    perts: List[str],
    y_true_mat: np.ndarray,
    y_pred_mat: np.ndarray,
    y_ctrl_vec: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate an array of predictions across multiple perturbations."""
    records = []
    for i, p in enumerate(perts):
        m = calculate_metrics(y_true_mat[i], y_pred_mat[i], y_ctrl_vec)
        m["perturbation"] = p
        records.append(m)

    df = pd.DataFrame(records)
    summary = {
        "pearson_all": float(df["pearson_all"].mean()),
        "mse_all": float(df["mse_all"].mean()),
        "pearson_top20_de": float(df["pearson_top20_de"].mean()),
        "mse_top20_de": float(df["mse_top20_de"].mean()),
        "n_perturbations": len(df),
    }
    return summary, df


def main():
    parser = argparse.ArgumentParser(description="Run cross-cell transfer benchmarks.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default="results/cross_cell", help="Output directory.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Compute Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Aligned Data and Precomputed ESM-2 Embeddings
    print("\n--- 1. Loading Aligned Data & ESM-2 Embeddings ---")
    aligned = CrossCellAlignedData.from_h5ad(cache_path="data/cross_cell_aligned_summary.npz")
    esm_dict = torch.load("data/embeddings/esm2_8m_cross_cell.pt", map_location="cpu", weights_only=False)
    print(f"Shared genes: {len(aligned.shared_genes):,}, Perturbations: {len(aligned.perturbations):,}")

    # 2. Split perturbations into Train, Val, OOD Test
    train_perts, val_perts, test_perts = aligned.split_perturbations(
        test_ratio=0.15, val_ratio=0.10, seed=args.seed
    )
    print(f"Splits: Train={len(train_perts)}, Val={len(val_perts)}, Test (OOD)={len(test_perts)}")

    train_idx = [aligned.pert2idx[p] for p in train_perts]
    val_idx = [aligned.pert2idx[p] for p in val_perts]
    test_idx = [aligned.pert2idx[p] for p in test_perts]

    # Pre-extract ESM features
    esm_matrix = np.array([esm_dict.get(p, np.zeros(320, dtype=np.float32)) for p in aligned.perturbations], dtype=np.float32)

    # Extract target gene basal expression in K562 and RPE1
    k562_basal = np.zeros((len(aligned.perturbations), 1), dtype=np.float32)
    rpe1_basal = np.zeros((len(aligned.perturbations), 1), dtype=np.float32)
    for i, p in enumerate(aligned.perturbations):
        if p in aligned.gene2idx:
            g_idx = aligned.gene2idx[p]
            k562_basal[i, 0] = aligned.k562_ctrl[g_idx]
            rpe1_basal[i, 0] = aligned.rpe1_ctrl[g_idx]

    # 3. Model Training
    print("\n--- 2. Training Models on K562 Essential ---")
    n_genes = len(aligned.shared_genes)

    # Tensor representations
    train_esm_t = torch.from_numpy(esm_matrix[train_idx]).to(device)
    train_ctrl_k562_t = torch.from_numpy(np.tile(aligned.k562_ctrl, (len(train_idx), 1))).to(device)
    train_basal_k562_t = torch.from_numpy(k562_basal[train_idx]).to(device)
    train_tgt_k562_t = torch.from_numpy(aligned.k562_means[train_idx]).to(device)

    val_esm_t = torch.from_numpy(esm_matrix[val_idx]).to(device)
    val_ctrl_k562_t = torch.from_numpy(np.tile(aligned.k562_ctrl, (len(val_idx), 1))).to(device)
    val_basal_k562_t = torch.from_numpy(k562_basal[val_idx]).to(device)
    val_tgt_k562_t = torch.from_numpy(aligned.k562_means[val_idx]).to(device)

    # Train BioTransferNet (De Novo OOD & Transfer Model)
    print("Training BioTransferNet...")
    biotransfer_model = BioTransferNet(n_genes=n_genes, esm_dim=320).to(device)
    opt_bt = AdamW(biotransfer_model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.MSELoss()

    ds = TensorDataset(train_esm_t, train_ctrl_k562_t, train_basal_k562_t, train_tgt_k562_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        biotransfer_model.train()
        for b_esm, b_ctrl, b_basal, b_tgt in loader:
            opt_bt.zero_grad()
            pred, _ = biotransfer_model(b_esm, b_ctrl, b_basal)
            loss = crit(pred, b_tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(biotransfer_model.parameters(), max_norm=1.0)
            opt_bt.step()

    # Train BioTransferTranslator (Screen Lineage Translation Model: K562 -> RPE1)
    print("Training BioTransferTranslator (K562 -> RPE1 CRISPR screen translator)...")
    translator_model = BioTransferTranslator(n_genes=n_genes, esm_dim=320).to(device)
    opt_tr = AdamW(translator_model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_delta_k562_t = torch.from_numpy(aligned.k562_deltas[train_idx]).to(device)
    train_delta_rpe1_t = torch.from_numpy(aligned.rpe1_deltas[train_idx]).to(device)
    train_ctrl_rpe1_t = torch.from_numpy(np.tile(aligned.rpe1_ctrl, (len(train_idx), 1))).to(device)

    ds_tr = TensorDataset(train_delta_k562_t, train_esm_t, train_ctrl_rpe1_t, train_delta_rpe1_t)
    loader_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        translator_model.train()
        for b_dk, b_esm, b_cr, b_dr in loader_tr:
            opt_tr.zero_grad()
            _, pred_delta = translator_model(b_dk, b_esm, b_cr)
            loss = crit(pred_delta, b_dr)
            loss.backward()
            nn.utils.clip_grad_norm_(translator_model.parameters(), max_norm=1.0)
            opt_tr.step()

    # Baseline: Empirical Mean Shift Vector on K562 Train
    k562_train_mean_shift = np.mean(aligned.k562_deltas[train_idx], axis=0)

    # 4. Multi-Task Evaluations
    print("\n--- 3. Running Multi-Task Evaluations ---")
    results = []

    eval_scenarios = [
        ("Task_1_K562_OOD_Test", test_perts, test_idx, "k562"),
        ("Task_2_RPE1_Transfer_TrainPerts", train_perts, train_idx, "rpe1"),
        ("Task_3_RPE1_Dual_ZeroShot_TestPerts", test_perts, test_idx, "rpe1"),
    ]

    for scenario_name, pert_list, idx_list, target_cell in eval_scenarios:
        target_ctrl = aligned.k562_ctrl if target_cell == "k562" else aligned.rpe1_ctrl
        y_true = aligned.k562_means[idx_list] if target_cell == "k562" else aligned.rpe1_means[idx_list]

        # 1. Control Mean
        pred_ctrl = np.tile(target_ctrl, (len(idx_list), 1))
        m_ctrl, _ = evaluate_predictions_batch(pert_list, y_true, pred_ctrl, target_ctrl)
        m_ctrl["Model"] = "Control Mean"
        m_ctrl["Scenario"] = scenario_name
        results.append(m_ctrl)

        # 2. Mean Shift Baseline (Ahlmann-Eltze 2025)
        pred_shift = np.tile(target_ctrl + k562_train_mean_shift, (len(idx_list), 1))
        m_shift, _ = evaluate_predictions_batch(pert_list, y_true, pred_shift, target_ctrl)
        m_shift["Model"] = "Mean Shift"
        m_shift["Scenario"] = scenario_name
        results.append(m_shift)

        # 3. Naive Cross-Cell Copying (applicable for RPE1 targets)
        if target_cell == "rpe1":
            pred_copy = target_ctrl[np.newaxis, :] + aligned.k562_deltas[idx_list]
            m_copy, df_copy_pert = evaluate_predictions_batch(pert_list, y_true, pred_copy, target_ctrl)
            m_copy["Model"] = "Naive K562 Shift Copy"
            m_copy["Scenario"] = scenario_name
            results.append(m_copy)

        # 4. BioTransferNet (De Novo OOD / Transfer)
        biotransfer_model.eval()
        with torch.no_grad():
            b_esm = torch.from_numpy(esm_matrix[idx_list]).to(device)
            b_ctrl = torch.from_numpy(np.tile(target_ctrl, (len(idx_list), 1))).to(device)
            b_basal = torch.from_numpy(k562_basal[idx_list] if target_cell == "k562" else rpe1_basal[idx_list]).to(device)
            pred_bt, _ = biotransfer_model(b_esm, b_ctrl, b_basal)
            pred_bt_np = pred_bt.cpu().numpy()

        m_bt, df_bt_pert = evaluate_predictions_batch(pert_list, y_true, pred_bt_np, target_ctrl)
        m_bt["Model"] = "BioTransferNet (De Novo)"
        m_bt["Scenario"] = scenario_name
        results.append(m_bt)

        # 5. BioTransferTranslator (Screen Lineage Translator) for RPE1
        if target_cell == "rpe1":
            translator_model.eval()
            with torch.no_grad():
                b_dk = torch.from_numpy(aligned.k562_deltas[idx_list]).to(device)
                b_esm = torch.from_numpy(esm_matrix[idx_list]).to(device)
                b_cr = torch.from_numpy(np.tile(target_ctrl, (len(idx_list), 1))).to(device)
                pred_tr, _ = translator_model(b_dk, b_esm, b_cr)
                pred_tr_np = pred_tr.cpu().numpy()

            m_tr, df_tr_pert = evaluate_predictions_batch(pert_list, y_true, pred_tr_np, target_ctrl)
            m_tr["Model"] = "BioTransferTranslator (Screen Translation)"
            m_tr["Scenario"] = scenario_name
            results.append(m_tr)

            if scenario_name == "Task_3_RPE1_Dual_ZeroShot_TestPerts":
                df_tr_pert.to_csv(out_dir / "dual_zero_shot_translator_breakdown.csv", index=False)
                df_copy_pert.to_csv(out_dir / "dual_zero_shot_naive_copy_breakdown.csv", index=False)

    results_df = pd.DataFrame(results)
    cols = ["Scenario", "Model", "pearson_top20_de", "mse_top20_de", "pearson_all", "mse_all", "n_perturbations"]
    results_df = results_df[cols]

    print("\n" + "=" * 105)
    print("                              UNIFIED BENCHMARK RESULTS SUMMARY")
    print("=" * 105)
    print(results_df.to_string(index=False))
    print("=" * 105)

    csv_path = out_dir / "cross_cell_benchmark_summary.csv"
    json_path = out_dir / "cross_cell_benchmark_summary.json"
    results_df.to_csv(csv_path, index=False)
    results_df.to_json(json_path, orient="records", indent=2)

    torch.save(
        {
            "biotransfer_net": biotransfer_model.state_dict(),
            "biotransfer_translator": translator_model.state_dict(),
            "shared_genes": aligned.shared_genes,
            "perturbations": aligned.perturbations,
            "train_perts": train_perts,
            "test_perts": test_perts,
        },
        out_dir / "biotransfer_models.pt",
    )
    print(f"\nSaved model weights and summaries to: {out_dir}")


if __name__ == "__main__":
    main()
