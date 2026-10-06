"""Neural baseline: Residual Multi-Layer Perceptron predicting expression shifts."""
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class PerturbationResidualMLP(nn.Module):
    """Neural network predicting post-perturbation cellular expression as a residual shift over control:

    ŷ = x̄_control + MLP(Embed(perturbation_idx))
    """

    def __init__(
        self,
        n_perturbations: int,
        n_genes: int,
        control_mean: np.ndarray,
        pert2idx: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 512],
        dropout: float = 0.1,
    ):
        """Initialize Residual MLP architecture.

        Parameters
        ----------
        n_perturbations : int
            Total number of unique perturbation classes.
        n_genes : int
            Total number of genes in expression space (output dimension).
        control_mean : np.ndarray
            Empirical control mean expression vector (length n_genes).
        pert2idx : dict
            Mapping from perturbation string to integer index.
        embedding_dim : int, default=128
            Dimension of the learned perturbation embeddings.
        hidden_dims : list of int, default=[256, 512]
            Hidden layer dimensions.
        dropout : float, default=0.1
            Dropout rate between dense layers.
        """
        super().__init__()
        self.n_perturbations = n_perturbations
        self.n_genes = n_genes
        self.pert2idx = pert2idx
        self.idx2pert = {i: p for p, i in pert2idx.items()}

        # Register non-trainable control mean buffer
        ctrl_tensor = torch.from_numpy(np.asarray(control_mean).astype(np.float32))
        self.register_buffer("control_mean", ctrl_tensor)

        # Perturbation embedding layer
        self.embedding = nn.Embedding(n_perturbations, embedding_dim)

        # Multi-layer perceptron projecting embedding to gene shift vector (Delta)
        layers: List[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final projection layer to gene dimension (initialized small for stable residual start)
        self.head = nn.Linear(in_dim, n_genes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.mlp = nn.Sequential(*layers)

    def forward(self, pert_indices: torch.Tensor) -> torch.Tensor:
        """Predict expression vectors for a batch of perturbation indices.

        Parameters
        ----------
        pert_indices : torch.Tensor
            1D LongTensor of perturbation indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Predicted expression tensor of shape (batch_size, n_genes).
        """
        embeds = self.embedding(pert_indices)
        deltas = self.head(self.mlp(embeds))
        # Add residual to unperturbed control profile
        predictions = self.control_mean.unsqueeze(0) + deltas
        # Clamp to non-negative log expression space
        return torch.clamp(predictions, min=0.0)

    @torch.no_grad()
    def predict(self, pert_name: str) -> np.ndarray:
        """Predict expression vector for a single perturbation by name.

        Implements interface required by BenchmarkHarness.
        """
        self.eval()
        if pert_name not in self.pert2idx:
            # Fall back to control mean if perturbation is unknown
            return self.control_mean.cpu().numpy().copy()

        idx = self.pert2idx[pert_name]
        idx_tensor = torch.tensor([idx], device=self.control_mean.device)
        pred = self.forward(idx_tensor)
        return pred.squeeze(0).cpu().numpy()
