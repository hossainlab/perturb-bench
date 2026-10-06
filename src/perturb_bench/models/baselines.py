"""Statistical and linear baselines for unseen perturbation response prediction."""
from typing import Dict, Optional
import numpy as np
import anndata as ad
from perturb_bench.data.loader import get_control_mean


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
        ctrl_mask = adata.obs["is_control"].astype(bool)
        train_obs = adata.obs.loc[(adata.obs["split"] == "train") & (~ctrl_mask)]
        train_perts = sorted(list(train_obs["perturbation"].unique()))

        if not train_perts:
            raise ValueError("No non-control training perturbations found in split == 'train'.")

        shifts = []
        for p in train_perts:
            sub = adata[(adata.obs["perturbation"] == p) & (adata.obs["split"] == "train")]
            p_mean = np.asarray(sub.X.mean(axis=0)).ravel()
            shifts.append(p_mean - self.control_mean)

        self.mean_shift = np.mean(shifts, axis=0)
        return self

    def predict(self, pert_name: str) -> np.ndarray:
        """Predict control mean plus average shift."""
        if self.control_mean is None or self.mean_shift is None:
            raise ValueError("Model must be fitted before predict() is called.")
        return self.control_mean + self.mean_shift
