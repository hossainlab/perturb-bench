"""Precompute ESM-2 embeddings for all perturbations in cross-cell aligned benchmark."""
from pathlib import Path
import numpy as np
import torch

from perturb_bench.data.transfer import CrossCellAlignedData
from perturb_bench.models.embeddings import (
    ESMProteinEmbedder,
    UniProtProteome,
    clean_perturbation_name,
)


def main():
    aligned_path = "data/cross_cell_aligned_summary.npz"
    proteome_path = "data/proteome/UP000005640_9606.fasta.gz"
    cache_path = "data/embeddings/esm2_8m_cross_cell.pt"

    print("Loading aligned data...")
    aligned = CrossCellAlignedData.from_h5ad(cache_path=aligned_path)
    print(f"Total perturbations to embed: {len(aligned.perturbations):,}")

    print("Loading UniProt proteome...")
    proteome = UniProtProteome(proteome_path)

    print("Initializing ESMProteinEmbedder...")
    embedder = ESMProteinEmbedder(
        model_name="facebook/esm2_t6_8M_UR50D",
        cache_path=cache_path,
        max_length=512,
    )

    # Compute in batches
    embedder.batch_embed_genes(aligned.perturbations, proteome, batch_size=32)

    # Compute mean embedding for any unmapped perturbations
    cached = embedder._cache
    valid_vectors = [v for k, v in cached.items() if k in aligned.perturbations]
    mean_vec = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(embedder.embedding_dim, dtype=np.float32)

    n_filled = 0
    for p in aligned.perturbations:
        if p not in cached:
            cached[p] = mean_vec
            n_filled += 1

    embedder.save_cache()
    print(f"\nFinal cache size: {len(cached):,} embeddings ({n_filled} filled with population mean).")
    print(f"Saved to: {cache_path}")


if __name__ == "__main__":
    main()
