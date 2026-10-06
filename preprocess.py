"""Preprocess scPerturb h5ad files into ready-to-train packages.

Pipeline used to produce the *_ready2train.h5ad files in this folder
(except replogle_k562_gwps, see preprocess_gwps.py).

Steps per dataset:
- load scPerturb h5ad (https://zenodo.org/records/13350497)
- dense -> sparse CSR
- raw-count detection -> normalize_total(1e4) + log1p if needed
- standardize obs: perturbation, is_control, cell_type (+ guide annotations)
- unseen-perturbation split 80/10/10 (seed 0); controls -> train
- write h5ad

Usage:
    python preprocess.py <scPerturb_file.h5ad> <output.h5ad> [--hvg N]
"""
import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 0


def looks_like_raw_counts(x):
    sample = x[:500]
    if sp.issparse(sample):
        sample = sample.toarray()
    mx = sample.max()
    pos = sample[sample > 0]
    frac_int = float(np.mean(pos == np.round(pos))) if pos.size else 0
    return mx > 25 and frac_int > 0.95


def find_pert_col(adata):
    for c in ['perturbation', 'gene', 'pert_id', 'guide']:
        if c in adata.obs.columns:
            return c
    raise KeyError(f'no perturbation column in {list(adata.obs.columns)}')


def is_control_mask(pert_series):
    p = pert_series.astype(str)
    return p.str.lower().isin(['non-targeting', 'ntc', 'control', 'nt_control',
                               'non-targeting-control', 'safe-targeting']) | p.str.startswith('NT')


def make_split(pert_values, is_ctrl, fracs=(0.8, 0.1, 0.1)):
    """Unseen-perturbation split: each perturbation wholly in one split; controls -> train."""
    rng = np.random.default_rng(SEED)
    perts = np.array(sorted({p for p, c in zip(pert_values, is_ctrl) if not c}))
    rng.shuffle(perts)
    n = len(perts)
    n_tr, n_va = int(fracs[0] * n), int(fracs[1] * n)
    assign = {p: 'train' for p in perts[:n_tr]}
    assign.update({p: 'val' for p in perts[n_tr:n_tr + n_va]})
    assign.update({p: 'test' for p in perts[n_tr + n_va:]})
    return np.array(['train' if c else assign.get(p, 'train')
                     for p, c in zip(pert_values, is_ctrl)], dtype=object)


def process(path, out_name, hvg_n=None):
    import anndata as ad
    name = path.split('/')[-1].replace('.h5ad', '')
    adata = ad.read_h5ad(path)
    print(f'--- {name}: {adata.shape}, X sparse={sp.issparse(adata.X)}', flush=True)

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    if looks_like_raw_counts(adata.X):
        import scanpy as sc
        tmp = ad.AnnData(X=adata.X)
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        adata.X = tmp.X
        print('  X: normalized_total(1e4)+log1p applied (source was raw counts)', flush=True)
    else:
        print('  X: source already log-normalized', flush=True)

    pcol = find_pert_col(adata)
    pert = adata.obs[pcol].astype(str).values
    ctrl = is_control_mask(pd.Series(pert)).values

    obs = pd.DataFrame(index=adata.obs_names.astype(str))
    obs['perturbation'] = pert
    obs['is_control'] = ctrl
    obs['cell_type'] = (adata.obs['cell_type'].astype(str).values
                        if 'cell_type' in adata.obs.columns else 'unknown')
    for extra in ['guide_id', 'guide_a', 'guide_b', 'npert_guides', 'gene_id']:
        if extra in adata.obs.columns:
            obs[extra] = adata.obs[extra].values

    var = pd.DataFrame(index=adata.var_names.astype(str))
    if hvg_n is not None and adata.n_vars > hvg_n:
        import scanpy as sc
        tmp = ad.AnnData(X=adata.X, obs=obs[['perturbation']], var=var)
        sc.pp.highly_variable_genes(tmp, n_top_genes=hvg_n)
        hvg_mask = tmp.var['highly_variable'].values
        pert_genes = {g for p in pert[~ctrl] for g in p.split('+')}
        pert_mask = var.index.isin(pert_genes & set(var.index))
        keep = hvg_mask | pert_mask
        adata = adata[:, keep]
        var = var.iloc[keep].copy()
        var['highly_variable'] = keep
        print(f'  genes: kept {int(keep.sum())} ({int(hvg_mask.sum())} HVG + perturbed genes)', flush=True)

    obs = obs.loc[adata.obs_names]
    obs['split'] = make_split(pert, ctrl)

    out = ad.AnnData(X=adata.X, obs=obs, var=var)
    out.write(out_name)
    n_pert = out.obs.loc[~out.obs['is_control'], 'perturbation'].nunique()
    sizes = out.obs['split'].value_counts().to_dict()
    print(f'SAVED {out_name}: {out.shape[0]} cells x {out.shape[1]} genes, '
          f'{n_pert} perturbations, splits={sizes}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--hvg', type=int, default=None,
                    help='subset to top-N HVGs (+ perturbed genes)')
    args = ap.parse_args()
    process(args.input, args.output, hvg_n=args.hvg)
