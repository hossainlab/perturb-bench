"""Unit tests for baseline predictors."""
import numpy as np
import pytest
from perturb_bench.data.loader import load_dataset
from perturb_bench.models.baselines import ControlMeanPredictor, MeanShiftPredictor
from perturb_bench.evaluation.benchmark import BenchmarkHarness


@pytest.fixture(scope="module")
def dixit_adata():
    return load_dataset("dixit")


def test_control_mean_predictor(dixit_adata):
    model = ControlMeanPredictor().fit(dixit_adata)
    assert model.control_mean is not None
    assert model.control_mean.shape == (dixit_adata.n_vars,)

    pred = model.predict("any_pert")
    assert pred.shape == (dixit_adata.n_vars,)
    assert np.allclose(pred, model.control_mean)


def test_mean_shift_predictor(dixit_adata):
    model = MeanShiftPredictor().fit(dixit_adata)
    assert model.mean_shift is not None
    assert model.mean_shift.shape == (dixit_adata.n_vars,)

    pred = model.predict("any_pert")
    assert pred.shape == (dixit_adata.n_vars,)
    # Shift should not be identical to control mean unless mean shift is all zeros
    assert not np.allclose(pred, model.control_mean)


def test_benchmark_harness_with_baselines(dixit_adata):
    harness = BenchmarkHarness(dixit_adata)
    assert len(harness.test_perts) > 0

    model = ControlMeanPredictor().fit(dixit_adata)
    summary, per_pert_df = harness.evaluate_model(model, model_name="CtrlMean")

    assert "Pearson ρ (All)" in summary
    assert "Pearson ρ (Top-20 DE)" in summary
    assert len(per_pert_df) == len(harness.test_perts)
    assert summary["Pearson ρ (All)"] > 0.8
