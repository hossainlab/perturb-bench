"""Unit tests for BioTransfer model architectures and embedding utilities."""
import numpy as np
import pytest
import torch

from perturb_bench.models.biotransfer import (
    BioTransferNet,
    BioTransferTranslator,
    ESMResidualMLP,
    IntegerResidualMLP,
)
from perturb_bench.models.embeddings import clean_perturbation_name


def test_clean_perturbation_name():
    assert clean_perturbation_name("p-sgIRF1-2") == ["IRF1"]
    assert clean_perturbation_name("OST4_pDS353") == ["OST4"]
    assert clean_perturbation_name("SET_KLF1") == ["SET", "KLF1"]
    assert clean_perturbation_name("control") == []
    assert clean_perturbation_name("non-targeting") == []
    assert clean_perturbation_name("RPL11") == ["RPL11"]


def test_biotransfer_net_forward_and_residual():
    n_genes = 100
    esm_dim = 32
    batch_size = 4

    model = BioTransferNet(n_genes=n_genes, esm_dim=esm_dim, hidden_dims=[64, 64])

    esm = torch.randn(batch_size, esm_dim)
    ctrl = torch.rand(batch_size, n_genes) + 0.1
    basal = torch.rand(batch_size, 1)

    pred_expr, pred_delta = model(esm, ctrl, basal)

    # Output shapes
    assert pred_expr.shape == (batch_size, n_genes)
    assert pred_delta.shape == (batch_size, n_genes)

    # Initial zero projection head guarantees Delta == 0 at initialization
    assert torch.allclose(pred_delta, torch.zeros_like(pred_delta))
    assert torch.allclose(pred_expr, ctrl)


def test_biotransfer_translator_forward():
    n_genes = 100
    esm_dim = 32
    batch_size = 4

    translator = BioTransferTranslator(n_genes=n_genes, esm_dim=esm_dim, hidden_dim=64)

    delta_src = torch.randn(batch_size, n_genes)
    esm = torch.randn(batch_size, esm_dim)
    ctrl_tgt = torch.rand(batch_size, n_genes) + 0.1

    pred_expr, pred_delta = translator(delta_src, esm, ctrl_tgt)

    assert pred_expr.shape == (batch_size, n_genes)
    assert pred_delta.shape == (batch_size, n_genes)
    # At initialization, correction head is 0, so Delta_tgt == Delta_src
    assert torch.allclose(pred_delta, delta_src)


def test_ablations_forward():
    n_genes = 50
    esm_dim = 16
    batch_size = 2

    m_esm = ESMResidualMLP(n_genes=n_genes, esm_dim=esm_dim, hidden_dims=[32])
    esm = torch.randn(batch_size, esm_dim)
    ctrl = torch.rand(batch_size, n_genes)
    expr_e, delta_e = m_esm(esm, ctrl)
    assert expr_e.shape == (batch_size, n_genes)

    m_int = IntegerResidualMLP(n_perturbations=10, n_genes=n_genes, embedding_dim=16, hidden_dims=[32])
    idx = torch.tensor([0, 1])
    expr_i, delta_i = m_int(idx, ctrl)
    assert expr_i.shape == (batch_size, n_genes)
