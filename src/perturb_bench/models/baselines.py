"""Statistical and linear baselines for unseen perturbation response prediction."""
from typing import Dict, Optional
import numpy as np
import anndata as ad
from perturb_bench.data.loader import get_control_mean, compute_group_means


class ControlMeanPredictor:
    """Baseline 1: Predicts the empirical mean control expression for every perturbation.

    ŷ_p = x̄_control
    """

    def __init__(self):
        self.control_mean: Optional[np.ndarray] = None

    def fit(self, adata: ad.AnnData) -> "ControlMeanPredictor":
        """Compute control mean profile from training cells."""
        self.control_mean = get_control_mean(adata)
        return self

    def predict(self, pert_name: str) -> np.ndarray:
        """Return the precomputed control profile."""
        if self.control_mean is None:
            raise ValueError("Model must be fitted before predict() is called.")
        return self.control_mean.copy()


class MeanShiftPredictor:
    """Baseline 2: Control mean plus global training mean shift.

    ŷ_p = x̄_control + Δ̄_train
    where Δ̄_train is the average shift across all training perturbations.
    """

    def __init__(self):
        self.control_mean: Optional[np.ndarray] = None
        self.mean_shift: Optional[np.ndarray] = None

    def fit(self, adata: ad.AnnData) -> "MeanShiftPredictor":
        """Fit control baseline and calculate mean training shift."""
        self.control_mean = get_control_mean(adata)

        # Compute shift for every training perturbation
        ctrl_mask = adata.obs["is_control"].astype(bool).values
        train_mask = (adata.obs["split"] == "train").values & (~ctrl_mask)
        sub_X = adata.X[train_mask]
        sub_perts = adata.obs.loc[train_mask, "perturbation"].values

        if len(sub_perts) == 0:
            raise ValueError("No non-control training perturbations found in split == 'train'.")

        _, group_means = compute_group_means(sub_X, sub_perts)
        shifts = group_means - self.control_mean
        self.mean_shift = np.mean(shifts, axis=0)
        return self

    def predict(self, pert_name: str) -> np.ndarray:
        """Predict control mean plus average shift."""
        if self.control_mean is None or self.mean_shift is None:
            raise ValueError("Model must be fitted before predict() is called.")
        return self.control_mean + self.mean_shift
