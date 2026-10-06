"""Chunked preprocessing for Replogle gwps (dense 2M x 8248 float32 on disk = ~145 GB).

Produces replogle_k562_gwps_ready2train.h5ad. The full gwps matrix cannot be
loaded into RAM on a single machine (145 GB dense), so this script streams it:

- Pass 1 (HVG): stream 100k random rows in chunks, log1p, accumulate per-gene
  mean/var -> top 5000 HVGs + all perturbed target genes.
- Cell subsampling: keep ALL controls + seeded random non-controls, sized so
  the final CSR fits comfortably in RAM (~<=15 GB).
- Pass 2: stream subsampled rows in chunks, slice genes, normalize
  (1e4 + log1p), collect CSR blocks, vstack, write h5ad with standardized
  obs + unseen-perturbation split (seed 0, same scheme as preprocess.py).

Usage:
    python preprocess_gwps.py <ReplogleWeissman2022_K562_gwps.h5ad> <output.h5ad>
"""
import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scipy.sparse as sp

SEED = 0
TARGET_CELLS = 500_000
HVG_N = 5000
MAX_CSR_GB = 15
CHUNK = 5000


def norm_chunk(x):
    lib = x.sum(1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(x / lib * 1e4)


def main(src, out_path):
    rng = np.random.default_rng(SEED)
    adata = ad.read_h5ad(src, backed='r')
    obs = adata.obs.copy()
    n, g = adata.shape
    pert = obs['gene'].astype(str).values  # scPerturb perturbation column
    ctrl = pd.Series(pert).str.lower().isin(
        ['non-targeting', 'ntc', 'control']).values | pd.Series(pert).str.startswith('NT').values
    print(f'gwps: {n} cells x {g} genes, {len(set(pert[~ctrl]))} perturbations, '
          f'{ctrl.sum()} control cells', flush=True)

    # ---- cell subsampling: all controls + random non-controls
    ctrl_idx = np.where(ctrl)[0]
    nonctrl_idx = np.where(~ctrl)[0]
    n_nonctrl = min(len(nonctrl_idx), TARGET_CELLS - len(ctrl_idx))
    sel_nonctrl = rng.choice(nonctrl_idx, size=n_nonctrl, replace=False)
    sel = np.sort(np.concatenate([ctrl_idx, sel_nonctrl]))
    print(f'subsample: {len(sel)} cells ({len(ctrl_idx)} controls + {n_nonctrl} perturbed)', flush=True)

    # ---- HVG pass on 100k random subsample of selection
    hvg_sample = np.sort(rng.choice(sel, size=min(100_000, len(sel)), replace=False))
    s1 = np.zeros(g); s2 = np.zeros(g); cnt = 0
    with h5py.File(src, 'r') as f:
        X = f['X']
        for i in range(0, len(hvg_sample), CHUNK):
            idx = hvg_sample[i:i + CHUNK]
            x = np.log1p(X[idx])  # raw counts -> log1p for variance ranking
            s1 += x.sum(0); s2 += (x ** 2).sum(0); cnt += len(idx)
    mean = s1 / cnt
    var = s2 / cnt - mean ** 2
    hvg_genes = np.argsort(var)[::-1][:HVG_N]
    print(f'HVG pass done on {cnt} cells', flush=True)

    # ---- gene set: HVGs + perturbed target genes
    pert_genes = {gt for p in pert[~ctrl] for gt in p.split('+')}
    var_names = np.array(adata.var_names.astype(str))
    pert_mask = np.isin(var_names, list(pert_genes & set(var_names)))
    gene_sel = np.sort(np.union1d(hvg_genes, np.where(pert_mask)[0]))
    print(f'gene selection: {len(gene_sel)} ({HVG_N} HVG + {int(pert_mask.sum())} perturbed)', flush=True)

    # ---- estimate density from one chunk to size the final matrix
    with h5py.File(src, 'r') as f:
        x0 = f['X'][sel[:CHUNK]][:, gene_sel]
    d0 = (x0 > 0).mean()
    est_gb = len(sel) * len(gene_sel) * d0 * 8 / 1e9
    print(f'density ~{d0:.3f} -> est CSR {est_gb:.1f} GB', flush=True)
    if est_gb > MAX_CSR_GB:
        keep_n = int(len(sel) * MAX_CSR_GB / est_gb)
        keep_n = min(keep_n, len(sel))
        extra = rng.choice(np.setdiff1d(sel, ctrl_idx), size=keep_n - len(ctrl_idx), replace=False)
        sel = np.sort(np.concatenate([ctrl_idx, np.intersect1d(extra, sel_nonctrl)]))
        print(f'trimmed to {len(sel)} cells to fit {MAX_CSR_GB} GB', flush=True)

    # ---- data pass: stream rows, collect CSR blocks
    blocks = []
    with h5py.File(src, 'r') as f:
        X = f['X']
        for i in range(0, len(sel), CHUNK):
            idx = sel[i:i + CHUNK]
            x = norm_chunk(X[idx][:, gene_sel].astype(np.float32))
            blocks.append(sp.csr_matrix(x))
            if (i // CHUNK) % 20 == 0:
                print(f'  row block {i}/{len(sel)}', flush=True)
    X_final = sp.vstack(blocks).tocsr()
    del blocks
    print(f'final X: {X_final.shape}, nnz={X_final.nnz} ({X_final.data.nbytes/1e9:.1f} GB data)', flush=True)

    # ---- obs/var/split
    sel_pert = pert[sel]
    sel_ctrl = ctrl[sel]
    obs_out = pd.DataFrame(index=adata.obs_names[sel].astype(str))
    obs_out['perturbation'] = sel_pert
    obs_out['is_control'] = sel_ctrl
    obs_out['cell_type'] = obs['cell_line'].astype(str).values[sel] if 'cell_line' in obs.columns else 'K562'
    if 'guide_id' in obs.columns:
        obs_out['guide_id'] = obs['guide_id'].astype(str).values[sel]
    obs_out['subsampled'] = True

    var_out = pd.DataFrame(index=var_names[gene_sel])
    var_out['highly_variable'] = np.isin(gene_sel, hvg_genes)

    # unseen-perturbation split (same logic as preprocess.py)
    perts_u = np.array(sorted({p for p, c in zip(sel_pert, sel_ctrl) if not c}))
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(perts_u)
    n_tr, n_va = int(0.8 * len(perts_u)), int(0.1 * len(perts_u))
    assign = {p: 'train' for p in perts_u[:n_tr]}
    assign.update({p: 'val' for p in perts_u[n_tr:n_tr + n_va]})
    assign.update({p: 'test' for p in perts_u[n_tr + n_va:]})
    obs_out['split'] = ['train' if c else assign.get(p, 'train') for p, c in zip(sel_pert, sel_ctrl)]

    out = ad.AnnData(X=X_final, obs=obs_out, var=var_out)
    out.write(out_path)
    sizes = out.obs['split'].value_counts().to_dict()
    n_pert = out.obs.loc[~out.obs['is_control'], 'perturbation'].nunique()
    print(f'SAVED {out_path}: {out.shape[0]} cells x {out.shape[1]} genes, '
          f'{n_pert} perturbations, splits={sizes}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    args = ap.parse_args()
    main(args.input, args.output)
