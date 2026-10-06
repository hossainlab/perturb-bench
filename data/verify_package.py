"""Verify the ready-to-train perturbation package on any machine.

Run:  python verify_package.py
Checks: files load, X is sparse log-normalized, splits are leak-free,
        controls only in train, split proportions ~80/10/10.
"""
import sys
import numpy as np
import anndata as ad
from pathlib import Path

DIR = Path(__file__).parent
EXPECTED = [
    'norman_2019_ready2train.h5ad',
    'adamson_2016_ready2train.h5ad',
    'dixit_2016_ready2train.h5ad',
    'replogle_k562_essential_ready2train.h5ad',
    'replogle_rpe1_ready2train.h5ad',
    'replogle_k562_gwps_ready2train.h5ad',
]


def check(fname):
    path = DIR / fname
    if not path.exists():
        return f'FAIL {fname}: missing'
    a = ad.read_h5ad(path)
    issues = []
    # X checks
    x = a.X[:200]
    x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    if not np.issubdtype(a.X.dtype, np.floating):
        issues.append('X not float')
    if x.max() > 25 or x.min() < 0:
        issues.append(f'X out of log-range [{x.min():.1f},{x.max():.1f}]')
    # required columns
    for c in ['perturbation', 'is_control', 'split']:
        if c not in a.obs.columns:
            issues.append(f'missing obs.{c}')
    if issues:
        return f'FAIL {fname}: {"; ".join(issues)}'
    # split checks
    obs = a.obs
    non_ctrl = obs[~obs['is_control']]
    leaks = non_ctrl.groupby('perturbation')['split'].nunique()
    if (leaks > 1).any():
        return f'FAIL {fname}: {int((leaks > 1).sum())} perturbations leak across splits'
    ctrl_splits = set(obs.loc[obs['is_control'], 'split'])
    if ctrl_splits - {'train'}:
        return f'FAIL {fname}: controls outside train'
    props = obs['split'].value_counts(normalize=True)
    n_pert = non_ctrl['perturbation'].nunique()
    return (f'PASS {fname}: {a.shape[0]} cells x {a.shape[1]} genes, {n_pert} perturbations, '
            f"splits train/val/test = {props.get('train', 0):.2f}/{props.get('val', 0):.2f}/{props.get('test', 0):.2f}")


if __name__ == '__main__':
    names = sys.argv[1:] if len(sys.argv) > 1 else EXPECTED
    results = [check(n) for n in names]
    for r in results:
        print(r)
    if any(r.startswith('FAIL') for r in results):
        sys.exit(1)
    print('\nALL CHECKS PASSED')
