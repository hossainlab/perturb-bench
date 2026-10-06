"""ESM-2 Protein Language Model Embeddings for Genetic Perturbations.

Provides continuous, biochemically grounded representations of perturbed proteins,
enabling zero-shot prediction of unseen gene knockouts and cross-cell-line transfer.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Common HGNC aliases and historical synonyms in Perturb-seq libraries
GENE_SYNONYMS: Dict[str, str] = {
    # tRNA synthetases (HGNC updated with 1)
    "AARS": "AARS1",
    "CARS": "CARS1",
    "DARS": "DARS1",
    "EPRS": "EPRS1",
    "GARS": "GARS1",
    "HARS": "HARS1",
    "IARS": "IARS1",
    "KARS": "KARS1",
    "LARS": "LARS1",
    "MARS": "MARS1",
    "NARS": "NARS1",
    "QARS": "QARS1",
    "RARS": "RARS1",
    "SARS": "SARS1",
    "TARS": "TARS1",
    "VARS": "VARS1",
    "WARS": "WARS1",
    "YARS": "YARS1",
    # Historical aliases & open reading frames
    "CD3EAP": "POLR1G",
    "C12ORF45": "C12orf45",
    "C14ORF178": "C14orf178",
    "C1ORF109": "C1orf109",
    "C7ORF26": "C7orf26",
    "C7ORF50": "C7orf50",
    "C9ORF16": "C9orf16",
    "FAM136A": "HYI",
    "FGFR1OP": "CEP43",
    "KIAA1804": "MAP3K21",
    "ELMSAN1": "MIDEAS",
    "MLL": "KMT2A",
    "MLL2": "KMT2D",
    "MLL4": "KMT2B",
    "TARDBP": "TARDBP",
}


def clean_perturbation_name(pert_str: str) -> List[str]:
    """Extract canonical target gene symbol(s) from dataset-specific perturbation labels.

    Examples:
    - 'p-sgIRF1-2' -> ['IRF1'] (Dixit)
    - 'OST4_pDS353' -> ['OST4'] (Adamson)
    - 'SET_KLF1' -> ['SET', 'KLF1'] (Norman dual)
    - 'RPL11' -> ['RPL11'] (Replogle)
    - 'control', 'non-targeting' -> []
    """
    if not isinstance(pert_str, str) or pert_str.lower() in {
        "control",
        "non-targeting",
        "intergenic",
        "nan",
        "*",
    }:
        return []

    # Adamson negative control plasmids
    if "(mod)_pba" in pert_str.lower():
        return []

    # Dixit format: p-sg<GENE>-<GUIDE_NUM>
    dixit_match = re.search(r"p-sg([A-Za-z0-9]+)-\d+", pert_str)
    if dixit_match:
        return [dixit_match.group(1).upper()]

    # Adamson format: <GENE>_pDS<GUIDE_NUM>
    adamson_match = re.search(r"^([A-Za-z0-9\-_]+)_pDS\d+", pert_str)
    if adamson_match:
        return [adamson_match.group(1).upper()]

    # Norman dual pairs: GENE1_GENE2
    if "_" in pert_str:
        parts = pert_str.split("_")
        return [p.strip().upper() for p in parts if p.strip()]

    return [pert_str.strip().upper()]


class UniProtProteome:
    """Parser and resolver for canonical human protein sequences from UniProt FASTA."""

    def __init__(self, fasta_path: Union[str, Path]):
        self.fasta_path = Path(fasta_path)
        self.gene_to_seq: Dict[str, str] = {}
        self.gene_to_accession: Dict[str, str] = {}
        self._load_fasta()

    def _load_fasta(self) -> None:
        """Parse gzipped UniProt reference proteome FASTA into memory."""
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"UniProt proteome file not found at: {self.fasta_path}")

        opener = gzip.open if self.fasta_path.suffix == ".gz" else open
        with opener(self.fasta_path, "rt", encoding="utf-8") as f:
            current_gene: Optional[str] = None
            current_acc: Optional[str] = None
            current_seq: List[str] = []

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_gene and current_seq:
                        self.gene_to_seq[current_gene] = "".join(current_seq)
                        if current_acc:
                            self.gene_to_accession[current_gene] = current_acc

                    # Extract primary accession and gene symbol
                    # Example: >sp|O14727|APAF_HUMAN ... GN=APAF1 ...
                    acc_match = re.search(r">[a-z]{2}\|([A-Z0-9]+)\|", line)
                    gn_match = re.search(r"GN=([A-Za-z0-9\-_]+)", line)

                    current_acc = acc_match.group(1) if acc_match else None
                    current_gene = gn_match.group(1).upper() if gn_match else None
                    current_seq = []
                else:
                    if current_gene:
                        current_seq.append(line)

            if current_gene and current_seq:
                self.gene_to_seq[current_gene] = "".join(current_seq)
                if current_acc:
                    self.gene_to_accession[current_gene] = current_acc

    def get_sequence(self, gene_symbol: str) -> Optional[str]:
        """Lookup canonical amino acid sequence by HGNC symbol with synonym resolution."""
        sym = gene_symbol.strip().upper()
        if sym in self.gene_to_seq:
            return self.gene_to_seq[sym]

        # Check synonym table
        if sym in GENE_SYNONYMS:
            syn = GENE_SYNONYMS[sym].upper()
            if syn in self.gene_to_seq:
                return self.gene_to_seq[syn]

        # Check if adding '1' matches (e.g. AARS -> AARS1)
        if f"{sym}1" in self.gene_to_seq:
            return self.gene_to_seq[f"{sym}1"]

        return None


class ESMProteinEmbedder:
    """Pretrained ESM-2 protein language model embedding extractor with local caching."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Optional[Union[str, torch.device]] = None,
        cache_path: Optional[Union[str, Path]] = None,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.cache_path = Path(cache_path) if cache_path else None

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._tokenizer = None
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Load precomputed embeddings from disk."""
        data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self._cache = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}

    def save_cache(self) -> None:
        """Persist computed embeddings to disk."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._cache, self.cache_path)

    def _init_model(self) -> None:
        """Lazy-load Hugging Face ESM-2 model onto compute device."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension of the underlying ESM model."""
        if "t6_8M" in self.model_name:
            return 320
        elif "t12_35M" in self.model_name:
            return 480
        elif "t30_150M" in self.model_name:
            return 640
        elif "t33_650M" in self.model_name:
            return 1280
        return 320

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Extract mean-pooled ESM-2 residue embedding for a protein sequence."""
        self._init_model()
        tokens = self._tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**tokens)
            # Mask special tokens when computing mean pool
            mask = tokens["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
            return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_gene(
        self,
        gene_symbol: str,
        proteome: UniProtProteome,
    ) -> Optional[np.ndarray]:
        """Retrieve or compute ESM embedding for a gene symbol."""
        gene_key = gene_symbol.strip().upper()
        if gene_key in self._cache:
            return self._cache[gene_key]

        seq = proteome.get_sequence(gene_key)
        if seq is None:
            return None

        emb = self.embed_sequence(seq)
        self._cache[gene_key] = emb
        return emb

    def batch_embed_genes(
        self,
        gene_symbols: List[str],
        proteome: UniProtProteome,
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """Precompute embeddings in batches for a list of gene symbols."""
        self._init_model()
        unique_genes = sorted(list(set(g.strip().upper() for g in gene_symbols if g)))
        to_compute = [g for g in unique_genes if g not in self._cache]

        print(f"Embedding {len(unique_genes)} genes ({len(to_compute)} new, {len(unique_genes) - len(to_compute)} cached)...")

        # Gather sequences
        valid_genes: List[str] = []
        valid_seqs: List[str] = []
        unmapped: List[str] = []

        for g in to_compute:
            seq = proteome.get_sequence(g)
            if seq:
                valid_genes.append(g)
                valid_seqs.append(seq)
            else:
                unmapped.append(g)

        if unmapped:
            print(f"Warning: {len(unmapped)} genes had no matching UniProt sequence.")

        # Batch forward passes
        for i in tqdm(range(0, len(valid_genes), batch_size), desc="Computing ESM-2 Embeddings"):
            batch_g = valid_genes[i : i + batch_size]
            batch_s = valid_seqs[i : i + batch_size]

            tokens = self._tokenizer(
                batch_s,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**tokens)
                mask = tokens["attention_mask"].unsqueeze(-1)
                hidden = outputs.last_hidden_state * mask
                pooled = (hidden.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)).cpu().numpy()

            for g, emb in zip(batch_g, pooled):
                self._cache[g] = emb.astype(np.float32)

        if self.cache_path:
            self.save_cache()
            print(f"Cached {len(self._cache)} protein embeddings to {self.cache_path}")

        return {g: self._cache[g] for g in unique_genes if g in self._cache}
