"""Unit tests for neural perturbation models."""
import torch
import numpy as np
import pytest
from perturb_bench.models.mlp import PerturbationResidualMLP


def test_perturbation_residual_mlp_forward():
    n_perts = 10
    n_genes = 50
    control_mean = np.random.uniform(0.1, 1.0, size=n_genes)
    pert2idx = {f"gene_{i}": i for i in range(n_perts)}

    model = PerturbationResidualMLP(
        n_perturbations=n_perts,
        n_genes=n_genes,
        control_mean=control_mean,
        pert2idx=pert2idx,
        embedding_dim=16,
        hidden_dims=[32, 64],
    )

    batch_indices = torch.tensor([0, 1, 3])
    out = model(batch_indices)

    assert out.shape == (3, n_genes)
    # Output should be non-negative
    assert (out >= 0.0).all()

    # Test predict() method
    pred = model.predict("gene_0")
    assert pred.shape == (n_genes,)
    assert isinstance(pred, np.ndarray)

    # Test unknown gene fallback to control mean
    pred_unknown = model.predict("unknown_gene")
    assert np.allclose(pred_unknown, control_mean, atol=1e-5)
