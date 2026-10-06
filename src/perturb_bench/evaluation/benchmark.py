"""Benchmark evaluation harness for evaluating perturbation response models."""
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
import pandas as pd
import anndata as ad
from perturb_bench.evaluation.metrics import (
    evaluate_perturbation,
    extract_top20_de_indices,
)
from perturb_bench.data.loader import get_control_mean, compute_group_means


class BenchmarkHarness:
    """Standardized evaluation harness running hold-out unseen test perturbation benchmarks."""

    def __init__(self, adata: ad.AnnData):
        """Precompute test set ground truth and differential expression masks.

        Parameters
        ----------
        adata : ad.AnnData
            Dataset with standardized `obs['split']` and `obs['is_control']`.
        """
        self.adata = adata
        self.n_genes = adata.n_vars
        self.gene_names = np.array(adata.var_names)

        # 1. Compute control mean
        self.control_mean = get_control_mean(adata)

        # 2. Extract hold-out test perturbations
        ctrl_mask = adata.obs["is_control"].astype(bool)
        test_obs = adata.obs.loc[(adata.obs["split"] == "test") & (~ctrl_mask)]
        self.test_perts = sorted(list(test_obs["perturbation"].unique()))

        if not self.test_perts:
            raise ValueError("No hold-out perturbations found in split == 'test' (excluding controls).")

        # 3. Vectorized precomputation of empirical test profiles
        ctrl_mask_arr = adata.obs["is_control"].astype(bool).values
        test_mask = (adata.obs["split"] == "test").values & (~ctrl_mask_arr)
        test_X = adata.X[test_mask]
        test_pert_labels = adata.obs.loc[test_mask, "perturbation"].values

        unique_perts, pert_means = compute_group_means(
            test_X, test_pert_labels, unique_groups=self.test_perts
        )
        pert_counts = pd.Series(test_pert_labels).value_counts().to_dict()

        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        for p, p_mean in zip(unique_perts, pert_means):
            top20_idx = extract_top20_de_indices(p_mean, self.control_mean, n_top=20)
            self.ground_truth[p] = {
                "mean_profile": p_mean,
                "n_cells": pert_counts.get(p, 0),
                "top20_de_idx": top20_idx,
            }

    def evaluate_model(
        self,
        model_or_predict_fn: Any,
        model_name: str = "Model",
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate a model or predict function on all hold-out test perturbations.

        Parameters
        ----------
        model_or_predict_fn : Any
            Either an object with a `predict(pert_name)` method, or a callable
            `predict_fn(pert_name) -> np.ndarray`.
        model_name : str, default="Model"
            Display name for the model in reports.

        Returns
        -------
        summary_dict : dict
            Averaged metrics across all test perturbations.
        per_pert_df : pd.DataFrame
            Detailed metrics for each individual test perturbation.
        """
        records: List[Dict[str, Any]] = []

        for p in self.test_perts:
            # Generate prediction
            if hasattr(model_or_predict_fn, "predict"):
                y_pred = model_or_predict_fn.predict(p)
            elif callable(model_or_predict_fn):
                y_pred = model_or_predict_fn(p)
            else:
                raise TypeError(
                    f"Model must have .predict() method or be a callable, got {type(model_or_predict_fn)}"
                )

            y_true = self.ground_truth[p]["mean_profile"]
            top20_idx = self.ground_truth[p]["top20_de_idx"]

            metrics = evaluate_perturbation(y_true, y_pred, top20_idx)
            metrics["perturbation"] = p
            metrics["n_cells"] = self.ground_truth[p]["n_cells"]
            records.append(metrics)

        per_pert_df = pd.DataFrame(records)

        summary = {
            "Model": model_name,
            "Pearson ρ (All)": float(per_pert_df["pearson_all"].mean()),
            "MSE (All)": float(per_pert_df["mse_all"].mean()),
            "Pearson ρ (Top-20 DE)": float(per_pert_df["pearson_de20"].mean()),
            "MSE (Top-20 DE)": float(per_pert_df["mse_de20"].mean()),
            "N Test Targets": len(self.test_perts),
        }

        return summary, per_pert_df

    def compare_models(
        self,
        models_dict: Dict[str, Any],
    ) -> pd.DataFrame:
        """Run benchmark comparison across multiple models and return summary table."""
        summaries = []
        for name, model in models_dict.items():
            summary, _ = self.evaluate_model(model, model_name=name)
            summaries.append(summary)
        return pd.DataFrame(summaries)
