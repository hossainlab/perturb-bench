"""Unit tests for benchmark metrics calculation."""
import numpy as np
import pytest
from perturb_bench.evaluation.metrics import (
    pearson_correlation,
    mean_squared_error,
    extract_top20_de_indices,
    evaluate_perturbation,
)


def test_pearson_correlation_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pearson_correlation(x, x)
    assert np.isclose(r, 1.0)


def test_pearson_correlation_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    r = pearson_correlation(x, y)
    assert np.isclose(r, -1.0)


def test_pearson_correlation_constant():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pearson_correlation(x, y)
    assert r == 0.0


def test_mean_squared_error():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    # diffs: 1, 2, 3 -> squares: 1, 4, 9 -> mean = 14 / 3 = 4.666...
    expected = (1.0 + 4.0 + 9.0) / 3.0
    assert np.isclose(mean_squared_error(x, y), expected)


def test_extract_top20_de_indices():
    ctrl = np.zeros(100)
    pert = np.zeros(100)
    # Set top 20 genes to high values
    pert[10:30] = 5.0

    top_idx = extract_top20_de_indices(pert, ctrl, n_top=20)
    assert len(top_idx) == 20
    assert set(top_idx) == set(range(10, 30))


def test_evaluate_perturbation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    top_idx = np.array([0, 1])

    res = evaluate_perturbation(y_true, y_pred, top_idx)
    assert "pearson_all" in res
    assert "mse_all" in res
    assert "pearson_de20" in res
    assert "mse_de20" in res
    assert res["pearson_all"] > 0.95
