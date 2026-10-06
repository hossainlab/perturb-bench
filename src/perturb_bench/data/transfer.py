"""Cross-Cell-Line Transfer Learning Data Alignment & Utilities.

Aligns K562 (myelogenous leukemia) and RPE1 (retinal pigment epithelial)
Perturb-seq datasets on shared gene expression space (7,226 genes) and shared
perturbations (2,055 gene knockouts) for zero-shot transfer modeling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from perturb_bench.data.loader import (
    compute_group_means,
    get_control_mean,
    load_dataset,
)


class CrossCellAlignedData:
    """Precomputed, memory-efficient container for paired cross-cell-line perturbation experiments.

    Contains:
    - Shared gene vocabulary (7,226 genes)
    - Shared perturbations (2,055 knockouts)
    - Empirical mean expression vectors for each perturbation in K562 and RPE1
    - Unperturbed control profiles for both cell lines
    - Differential expression shifts: Delta = mean_p - mean_ctrl
    """

    def __init__(
        self,
        shared_genes: List[str],
        perturbations: List[str],
        k562_means: np.ndarray,
        rpe1_means: np.ndarray,
        k562_ctrl: np.ndarray,
        rpe1_ctrl: np.ndarray,
        pert_cell_counts_k562: Dict[str, int],
        pert_cell_counts_rpe1: Dict[str, int],
    ):
        self.shared_genes = shared_genes
        self.gene2idx = {g: i for i, g in enumerate(shared_genes)}
        self.perturbations = perturbations
        self.pert2idx = {p: i for i, p in enumerate(perturbations)}

        self.k562_means = k562_means.astype(np.float32)  # shape: (N_perts, G)
        self.rpe1_means = rpe1_means.astype(np.float32)  # shape: (N_perts, G)
        self.k562_ctrl = k562_ctrl.astype(np.float32)    # shape: (G,)
        self.rpe1_ctrl = rpe1_ctrl.astype(np.float32)    # shape: (G,)

        # Precompute differential shifts
        self.k562_deltas = self.k562_means - self.k562_ctrl[np.newaxis, :]
        self.rpe1_deltas = self.rpe1_means - self.rpe1_ctrl[np.newaxis, :]

        self.pert_cell_counts_k562 = pert_cell_counts_k562
        self.pert_cell_counts_rpe1 = pert_cell_counts_rpe1

    @classmethod
    def from_h5ad(
        cls,
        data_dir: Optional[Union[str, Path]] = None,
        cache_path: Optional[Union[str, Path]] = "data/cross_cell_aligned_summary.npz",
    ) -> CrossCellAlignedData:
        """Load from precomputed cache or compute directly from raw H5AD files."""
        if cache_path:
            p = Path(cache_path)
            if p.exists():
                print(f"Loading cached cross-cell aligned data from {p}...")
                data = np.load(p, allow_pickle=True)
                return cls(
                    shared_genes=list(data["shared_genes"]),
                    perturbations=list(data["perturbations"]),
                    k562_means=data["k562_means"],
                    rpe1_means=data["rpe1_means"],
                    k562_ctrl=data["k562_ctrl"],
                    rpe1_ctrl=data["rpe1_ctrl"],
                    pert_cell_counts_k562=data["pert_counts_k562"].item(),
                    pert_cell_counts_rpe1=data["pert_counts_rpe1"].item(),
                )

        print("Aligning K562 Essential and RPE1 Essential datasets...")
        k562_adata = load_dataset("replogle_k562_essential", data_dir=data_dir)
        rpe1_adata = load_dataset("replogle_rpe1", data_dir=data_dir)

        # 1. Intersect genes
        shared_genes = sorted(list(set(k562_adata.var_names).intersection(set(rpe1_adata.var_names))))
        print(f"Identified {len(shared_genes):,} shared genes.")

        k562_gene_indices = [k562_adata.var_names.get_loc(g) for g in shared_genes]
        rpe1_gene_indices = [rpe1_adata.var_names.get_loc(g) for g in shared_genes]

        # 2. Intersect perturbations (excluding controls)
        k562_perts = set(k562_adata.obs["perturbation"].dropna().unique()) - {"control", "non-targeting"}
        rpe1_perts = set(rpe1_adata.obs["perturbation"].dropna().unique()) - {"control", "non-targeting"}
        shared_perts = sorted(list(k562_perts.intersection(rpe1_perts)))
        print(f"Identified {len(shared_perts):,} shared perturbations.")

        # 3. Compute control means on shared gene space
        k562_ctrl_full = get_control_mean(k562_adata)
        rpe1_ctrl_full = get_control_mean(rpe1_adata)
        k562_ctrl = k562_ctrl_full[k562_gene_indices]
        rpe1_ctrl = rpe1_ctrl_full[rpe1_gene_indices]

        # 4. Compute vectorized perturbation group means
        print("Computing vectorized perturbation means for K562...")
        k562_perts_list, k562_means_arr = compute_group_means(k562_adata.X, k562_adata.obs["perturbation"].values)
        k562_means_dict = dict(zip(k562_perts_list, k562_means_arr))

        print("Computing vectorized perturbation means for RPE1...")
        rpe1_perts_list, rpe1_means_arr = compute_group_means(rpe1_adata.X, rpe1_adata.obs["perturbation"].values)
        rpe1_means_dict = dict(zip(rpe1_perts_list, rpe1_means_arr))

        k562_means_mat = np.zeros((len(shared_perts), len(shared_genes)), dtype=np.float32)
        rpe1_means_mat = np.zeros((len(shared_perts), len(shared_genes)), dtype=np.float32)

        counts_k562 = k562_adata.obs["perturbation"].value_counts().to_dict()
        counts_rpe1 = rpe1_adata.obs["perturbation"].value_counts().to_dict()

        pert_counts_k562 = {}
        pert_counts_rpe1 = {}

        for i, p in enumerate(shared_perts):
            k562_means_mat[i] = k562_means_dict[p][k562_gene_indices]
            rpe1_means_mat[i] = rpe1_means_dict[p][rpe1_gene_indices]
            pert_counts_k562[p] = int(counts_k562.get(p, 0))
            pert_counts_rpe1[p] = int(counts_rpe1.get(p, 0))

        aligned = cls(
            shared_genes=shared_genes,
            perturbations=shared_perts,
            k562_means=k562_means_mat,
            rpe1_means=rpe1_means_mat,
            k562_ctrl=k562_ctrl,
            rpe1_ctrl=rpe1_ctrl,
            pert_cell_counts_k562=pert_counts_k562,
            pert_cell_counts_rpe1=pert_counts_rpe1,
        )

        if cache_path:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            print(f"Caching aligned cross-cell summary to {p}...")
            np.savez_compressed(
                p,
                shared_genes=np.array(shared_genes),
                perturbations=np.array(shared_perts),
                k562_means=k562_means_mat,
                rpe1_means=rpe1_means_mat,
                k562_ctrl=k562_ctrl,
                rpe1_ctrl=rpe1_ctrl,
                pert_counts_k562=pert_counts_k562,
                pert_counts_rpe1=pert_counts_rpe1,
            )

        return aligned

    def split_perturbations(
        self,
        test_ratio: float = 0.15,
        val_ratio: float = 0.10,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split shared perturbations into train, validation, and hold-out test sets.

        The test set represents unseen out-of-distribution gene knockouts for evaluation.
        """
        rng = np.random.default_rng(seed)
        shuffled = np.array(self.perturbations.copy())
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_perts = list(shuffled[:n_test])
        val_perts = list(shuffled[n_test : n_test + n_val])
        train_perts = list(shuffled[n_test + n_val :])

        return train_perts, val_perts, test_perts


class CrossCellDataset(Dataset):
    """PyTorch Dataset for paired perturbation training and cross-cell transfer."""

    def __init__(
        self,
        aligned_data: CrossCellAlignedData,
        perturbation_list: List[str],
        protein_embeddings: Dict[str, np.ndarray],
        cell_line: str = "k562",
    ):
        self.aligned = aligned_data
        self.perts = [p for p in perturbation_list if p in aligned_data.pert2idx and p in protein_embeddings]
        self.protein_embeddings = protein_embeddings
        self.cell_line = cell_line.lower()

        # Select corresponding source arrays
        if self.cell_line == "k562":
            self.means = aligned_data.k562_means
            self.deltas = aligned_data.k562_deltas
            self.ctrl = torch.from_numpy(aligned_data.k562_ctrl)
        elif self.cell_line == "rpe1":
            self.means = aligned_data.rpe1_means
            self.deltas = aligned_data.rpe1_deltas
            self.ctrl = torch.from_numpy(aligned_data.rpe1_ctrl)
        else:
            raise ValueError(f"Unknown cell line: {cell_line}")

        self.pert_indices = [aligned_data.pert2idx[p] for p in self.perts]

    def __len__(self) -> int:
        return len(self.perts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pert_name = self.perts[idx]
        mat_idx = self.pert_indices[idx]

        emb = torch.from_numpy(self.protein_embeddings[pert_name])
        target_expr = torch.from_numpy(self.means[mat_idx])

        return emb, target_expr, pert_name
