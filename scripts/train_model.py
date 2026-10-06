#!/usr/bin/env python
"""CLI tool to train a neural perturbation response model on Apple Silicon / CUDA."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from perturb_bench.data.dataset import create_dataloaders
from perturb_bench.data.loader import get_control_mean, load_dataset
from perturb_bench.evaluation.benchmark import BenchmarkHarness
from perturb_bench.models.mlp import PerturbationResidualMLP
from perturb_bench.utils.device import get_device
from perturb_bench.utils.seed import set_seed


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_pert_idx, _ in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_pert_idx = batch_pert_idx.to(device, non_blocking=True)

        optimizer.zero_grad()
        predictions = model(batch_pert_idx)
        loss = criterion(predictions, batch_x)
        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_pert_idx, _ in data_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_pert_idx = batch_pert_idx.to(device, non_blocking=True)
        predictions = model(batch_pert_idx)
        loss = criterion(predictions, batch_x)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser(description="Train PerturbationResidualMLP model.")
    parser.add_argument("--dataset", type=str, default="dixit", help="Dataset name ('dixit', 'adamson', 'norman').")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (64-256 recommended for 24 GB).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimension of perturbation embeddings.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 512], help="Hidden layer dimensions.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cuda', 'cpu').")
    parser.add_argument("--out-dir", type=str, default="results/checkpoints", help="Output directory for checkpoints.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Device selection
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using compute device: {device}")

    # 2. Data loading
    print(f"\nLoading dataset '{args.dataset}'...")
    adata = load_dataset(args.dataset)
    print(f"Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    ctrl_mean = get_control_mean(adata)
    train_loader, val_loader, test_loader, pert2idx, _ = create_dataloaders(
        adata, batch_size=args.batch_size
    )
    print(f"Unique perturbations in vocabulary: {len(pert2idx):,}")
    print(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # 3. Model instantiation
    model = PerturbationResidualMLP(
        n_perturbations=len(pert2idx),
        n_genes=adata.n_vars,
        control_mean=ctrl_mean,
        pert2idx=pert2idx,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized: {n_params:,} trainable parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 4. Training loop
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = out_dir / "best_model.pt"

    best_val_loss = float("inf")
    history = []

    print(f"\n--- Beginning Training ({args.epochs} epochs) ---")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step()
        ep_time = time.time() - ep_t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "pert2idx": pert2idx,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

        star = " *" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {ep_time:.1f}s{star}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        })

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.5f}")

    # 5. Evaluate best model using BenchmarkHarness
    print("\n--- Evaluating Best Model on Hold-Out Test Set ---")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    harness = BenchmarkHarness(adata)
    test_summary, per_pert_df = harness.evaluate_model(model, model_name=f"ResidualMLP ({args.dataset})")

    print("\nHOLD-OUT TEST EVALUATION METRICS:")
    for k, v in test_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

    # 6. Save logs and metrics
    eval_path = out_dir / "test_evaluation.json"
    history_path = out_dir / "training_history.json"
    per_pert_path = out_dir / "per_perturbation_metrics.csv"

    with open(eval_path, "w") as f:
        json.dump(test_summary, f, indent=2)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    per_pert_df.to_csv(per_pert_path, index=False)

    print(f"\nArtifacts saved to {out_dir}:")
    print(f"  - Model: {best_checkpoint_path.name}")
    print(f"  - Test Summary: {eval_path.name}")
    print(f"  - Training History: {history_path.name}")
    print(f"  - Per-Target Breakdown: {per_pert_path.name}")


if __name__ == "__main__":
    main()
