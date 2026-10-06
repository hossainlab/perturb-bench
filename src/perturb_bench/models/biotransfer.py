"""BioTransfer: Cell-Conditioned Multimodal Architecture for Perturbation Prediction.

Integrates ESM-2 evolutionary protein language model priors with cell-state
transcriptomic conditioning to enable:
1. Zero-shot prediction of unseen gene knockouts (out-of-distribution generalization).
2. Zero-shot cross-cell-line transfer (e.g., K562 -> RPE1).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BioTransferNet(nn.Module):
    """Cell-Conditioned Residual Network with Protein Language Model Priors.

    Formulation:
        z_p = MLP_pert([ESM(p) || x_ctrl[p]])
        c_cell = MLP_cell(x_ctrl)
        gamma, beta = FiLM(c_cell)
        h = (1 + gamma) * z_p + beta
        Delta = ZeroInitHead(h)
        x_hat = clamp(x_ctrl + Delta, min=0.0)
    """

    def __init__(
        self,
        n_genes: int,
        esm_dim: int = 320,
        cell_latent_dim: int = 128,
        hidden_dims: List[int] = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.esm_dim = esm_dim
        self.cell_latent_dim = cell_latent_dim

        # 1. Perturbation feature encoder (ESM-2 embedding + 1 basal target gene feature)
        self.pert_encoder = nn.Sequential(
            nn.Linear(esm_dim + 1, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Cell context encoder (compresses full basal transcriptome profile)
        self.cell_encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, cell_latent_dim),
            nn.LayerNorm(cell_latent_dim),
        )

        # 3. FiLM (Feature-wise Linear Modulation) generator: cell state modulates perturbation
        self.film_gamma = nn.Linear(cell_latent_dim, hidden_dims[0])
        self.film_beta = nn.Linear(cell_latent_dim, hidden_dims[0])
        # Initialize FiLM to identity transform
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # 4. Deep interaction trunk
        trunk_layers: List[nn.Module] = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            trunk_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()

        # 5. Zero-initialized residual head predicting delta shifts
        self.head = nn.Linear(in_dim, n_genes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        ctrl_expression: torch.Tensor,
        target_gene_basal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting differential shifts and final expression.

        Parameters
        ----------
        esm_embeddings : torch.Tensor
            Batch of ESM-2 embeddings of shape (batch_size, esm_dim).
        ctrl_expression : torch.Tensor
            Unperturbed control profile of target cell line (batch_size, n_genes) or (1, n_genes).
        target_gene_basal : torch.Tensor
            Basal expression scalar for the perturbed target gene (batch_size, 1).

        Returns
        -------
        pred_expression : torch.Tensor
            Predicted post-perturbation expression of shape (batch_size, n_genes).
        pred_delta : torch.Tensor
            Predicted differential shift Delta of shape (batch_size, n_genes).
        """
        # Encode perturbation
        pert_in = torch.cat([esm_embeddings, target_gene_basal], dim=-1)
        h_pert = self.pert_encoder(pert_in)

        # Encode cell context
        c_cell = self.cell_encoder(ctrl_expression)

        # FiLM modulation
        gamma = self.film_gamma(c_cell)
        beta = self.film_beta(c_cell)
        h_modulated = (1.0 + gamma) * h_pert + beta

        # Process through trunk and project
        h_out = self.trunk(h_modulated)
        pred_delta = self.head(h_out)

        # Residual combination with control profile
        pred_expression = torch.clamp(ctrl_expression + pred_delta, min=0.0)
        return pred_expression, pred_delta


class ESMResidualMLP(nn.Module):
    """Ablation Model: ESM-2 embeddings without cell-context conditioning."""

    def __init__(
        self,
        n_genes: int,
        esm_dim: int = 320,
        hidden_dims: List[int] = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.esm_dim = esm_dim

        layers: List[nn.Module] = []
        in_dim = esm_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_genes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        ctrl_expression: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_delta = self.head(self.mlp(esm_embeddings))
        pred_expression = torch.clamp(ctrl_expression + pred_delta, min=0.0)
        return pred_expression, pred_delta


class IntegerResidualMLP(nn.Module):
    """Ablation Model: Standard integer lookup embedding (no protein prior)."""

    def __init__(
        self,
        n_perturbations: int,
        n_genes: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_perturbations = n_perturbations
        self.n_genes = n_genes

        self.embedding = nn.Embedding(n_perturbations, embedding_dim)
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

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_genes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pert_indices: torch.Tensor,
        ctrl_expression: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(pert_indices)
        pred_delta = self.head(self.mlp(embeds))
        pred_expression = torch.clamp(ctrl_expression + pred_delta, min=0.0)
        return pred_expression, pred_delta


class BioTransferTranslator(nn.Module):
    """Cross-Cell Lineage Translation Network.

    Translates single-cell perturbation differential shifts from a high-throughput
    source cell line (e.g. suspension K562) to a target cell line (e.g. adherent diploid RPE1)
    using ESM-2 protein language representations and target cell basal conditioning.

    Formulation:
        gate_p = Sigmoid(MLP_gate(ESM(p)))
        Delta_correction = MLP_trans(Delta_source)
        Delta_target_pred = Delta_source + gate_p * Delta_correction
        x_target_pred = clamp(x_ctrl_target + Delta_target_pred, min=0.0)
    """

    def __init__(
        self,
        n_genes: int,
        esm_dim: int = 320,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.esm_dim = esm_dim

        # Protein-specific gating network
        self.pert_gate = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
            nn.Sigmoid(),
        )

        # Cross-cell shift transformation trunk
        self.delta_transform = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )
        # Zero-initialize the transformation head so training starts from identity copy
        nn.init.zeros_(self.delta_transform[-1].weight)
        nn.init.zeros_(self.delta_transform[-1].bias)

    def forward(
        self,
        delta_source: torch.Tensor,
        esm_embeddings: torch.Tensor,
        ctrl_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.pert_gate(esm_embeddings)
        correction = self.delta_transform(delta_source)
        pred_delta_target = delta_source + gate * correction
        pred_expr_target = torch.clamp(ctrl_target + pred_delta_target, min=0.0)
        return pred_expr_target, pred_delta_target

