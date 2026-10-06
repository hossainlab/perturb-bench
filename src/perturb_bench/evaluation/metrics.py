"""Benchmark evaluation metrics: Pearson correlation and MSE overall and on top-20 DE genes."""
from typing import Dict, Union
import numpy as np


def pearson_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8
) -> float:
    """Compute Pearson correlation coefficient between two 1D expression vectors.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth post-perturbation expression vector (length G).
    y_pred : np.ndarray
        Predicted post-perturbation expression vector (length G).
    eps : float, default=1e-8
        Numerical tolerance for zero standard deviation.

    Returns
    -------
    float
        Pearson correlation in [-1.0, 1.0]. Returns 0.0 if either vector is constant.
    """
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < eps or std_pred < eps:
        return 0.0
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(np.nan_to_num(r, nan=0.0))


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute Mean Squared Error (MSE) between two 1D expression vectors."""
    return float(np.mean((y_true - y_pred) ** 2))


def extract_top20_de_indices(
    true_pert_mean: np.ndarray,
    control_mean: np.ndarray,
    n_top: int = 20
) -> np.ndarray:
    """Identify the indices of the top-N differential expression genes."""
    delta = np.abs(true_pert_mean - control_mean)
    return np.argsort(delta)[::-1][:n_top]


def evaluate_perturbation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_de_indices: np.ndarray,
) -> Dict[str, float]:
    """Compute standard benchmark metrics on all genes and on top DE genes.

    Returns
    -------
    dict
        Dictionary containing:
        - 'pearson_all': Pearson correlation across all genes
        - 'mse_all': MSE across all genes
        - 'pearson_de20': Pearson correlation restricted to top-20 DE genes
        - 'mse_de20': MSE restricted to top-20 DE genes
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r_all = pearson_correlation(y_true, y_pred)
    mse_all = mean_squared_error(y_true, y_pred)

    y_true_de = y_true[top_de_indices]
    y_pred_de = y_pred[top_de_indices]

    r_de = pearson_correlation(y_true_de, y_pred_de)
    mse_de = mean_squared_error(y_true_de, y_pred_de)

    return {
        "pearson_all": r_all,
        "mse_all": mse_all,
        "pearson_de20": r_de,
        "mse_de20": mse_de,
    }
