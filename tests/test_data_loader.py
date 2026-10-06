"""Unit tests for dataset loading and DataLoader construction."""
import torch
import pytest
from perturb_bench.data.loader import load_dataset, get_control_mean
from perturb_bench.data.dataset import create_dataloaders, PerturbationDataset


def test_load_dixit():
    adata = load_dataset("dixit")
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert "is_control" in adata.obs.columns
    assert "split" in adata.obs.columns

    # Verify controls exist
    ctrl_mean = get_control_mean(adata)
    assert ctrl_mean.shape == (adata.n_vars,)
    assert (ctrl_mean >= 0.0).all()


def test_dataloaders_generation():
    adata = load_dataset("dixit")
    train_loader, val_loader, test_loader, pert2idx, idx2pert = create_dataloaders(
        adata, batch_size=64
    )

    assert len(pert2idx) > 0
    assert len(train_loader) > 0

    batch_x, batch_idx, batch_name = next(iter(train_loader))
    assert batch_x.shape[0] <= 64
    assert batch_x.shape[1] == adata.n_vars
    assert batch_x.dtype == torch.float32
    assert isinstance(batch_name[0], str)
