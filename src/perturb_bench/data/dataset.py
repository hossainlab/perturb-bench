"""Memory-safe PyTorch Dataset and DataLoader generators."""
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad


class PerturbationDataset(Dataset):
    """Memory-safe PyTorch Dataset preserving sparse CSR matrices in RAM.

    Each item is densified only when requested by DataLoader collator,
    preventing RAM exhaustion on large perturbation screens.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        split: str,
        pert2idx: Optional[Dict[str, int]] = None,
    ):
        """Initialize split subset.

        Parameters
        ----------
        adata : ad.AnnData
            Full AnnData object.
        split : str
            One of 'train', 'val', or 'test'.
        pert2idx : dict, optional
            Mapping from perturbation string to integer index. If None,
            built from unique perturbations in this split.
        """
        mask = (adata.obs["split"] == split).values
        self.sub_adata = adata[mask]
        self.split = split
        self.perts = self.sub_adata.obs["perturbation"].values
        self.is_control = self.sub_adata.obs["is_control"].astype(bool).values

        # Build or assign vocabulary
        if pert2idx is None:
            unique_perts = sorted(list(set(self.perts)))
            self.pert2idx = {p: i for i, p in enumerate(unique_perts)}
        else:
            self.pert2idx = pert2idx
        self.idx2pert = {i: p for p, i in self.pert2idx.items()}

    def __len__(self) -> int:
        return len(self.sub_adata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.sub_adata.X[idx]
        x_dense = np.asarray(row.todense() if sp.issparse(row) else row).ravel()
        x_tensor = torch.from_numpy(x_dense.astype(np.float32))

        pert_name = self.perts[idx]
        pert_idx = self.pert2idx.get(pert_name, -1)

        return x_tensor, pert_idx, pert_name


def create_dataloaders(
    adata: ad.AnnData,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Build unified vocabulary and return train, val, and test DataLoaders.

    Parameters
    ----------
    adata : ad.AnnData
        Preprocessed AnnData object.
    batch_size : int, default=128
        Batch size (64-256 recommended for 24 GB unified memory).
    num_workers : int, default=0
        Number of worker threads (0 recommended for Apple Silicon unified memory).

    Returns
    -------
    train_loader, val_loader, test_loader, pert2idx, idx2pert
    """
    # Build complete vocabulary over all unique perturbations
    all_unique_perts = sorted(list(adata.obs["perturbation"].unique()))
    pert2idx = {p: i for i, p in enumerate(all_unique_perts)}
    idx2pert = {i: p for p, i in pert2idx.items()}

    train_ds = PerturbationDataset(adata, "train", pert2idx=pert2idx)
    val_ds = PerturbationDataset(adata, "val", pert2idx=pert2idx)
    test_ds = PerturbationDataset(adata, "test", pert2idx=pert2idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, pert2idx, idx2pert
