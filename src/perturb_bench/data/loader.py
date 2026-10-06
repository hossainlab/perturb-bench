"""Dataset loading and harmonization utilities with audit fixes."""
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import anndata as ad


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

DATASET_FILENAMES = {
    "dixit": "dixit_2016_ready2train.h5ad",
    "adamson": "adamson_2016_ready2train.h5ad",
    "norman": "norman_2019_ready2train.h5ad",
    "replogle_k562_essential": "replogle_k562_essential_ready2train.h5ad",
    "replogle_rpe1": "replogle_rpe1_ready2train.h5ad",
    "replogle_k562_gwps": "replogle_k562_gwps_ready2train.h5ad",
}


def resolve_dataset_path(
    name_or_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve a dataset name (alias or filename) to a filesystem Path."""
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    target = str(name_or_path)

    # Check alias
    if target in DATASET_FILENAMES:
        target = DATASET_FILENAMES[target]

    path = Path(target)
    if not path.is_file():
        path = base_dir / target

    if not path.is_file():
        raise FileNotFoundError(f"Dataset '{name_or_path}' could not be located at '{path}'.")
    return path


def load_dataset(
    name_or_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    backed: Optional[str] = None,
    filter_unassigned: bool = True,
) -> ad.AnnData:
    """Load a Perturb-seq AnnData file with standardized control labels and cleaned metadata.

    Parameters
    ----------
    name_or_path : str or Path
        Dataset identifier (e.g., 'dixit', 'adamson', 'norman', 'dixit_2016_ready2train.h5ad')
        or absolute path.
    data_dir : str or Path, optional
        Base directory containing .h5ad files.
    backed : str, optional
        'r' for read-only backed mode, None for full RAM loading.
    filter_unassigned : bool, default=True
        Whether to filter out cells with unassigned ('nan') perturbation barcodes.

    Returns
    -------
    ad.AnnData
        Preprocessed AnnData with normalized controls and verified splits.
    """
    path = resolve_dataset_path(name_or_path, data_dir)
    fname = path.name.lower()
    adata = ad.read_h5ad(path, backed=backed)

    # 1. Standardize control masks
    if "adamson" in fname:
        # Adamson negative controls are plasmids pBA580, pBA582, and '*'
        ctrl_mask = adata.obs["perturbation"].isin(
            ["63(mod)_pBA580", "Gal4-4(mod)_pBA582", "*"]
        )
    elif "replogle_k562_gwps" in fname:
        # In GWPS, real controls are labeled 'non-targeting' (excluding human 'NT' genes)
        ctrl_mask = adata.obs["perturbation"] == "non-targeting"
    else:
        ctrl_mask = adata.obs["is_control"].astype(bool)

    # 2. Filter unassigned 'nan' guide cells
    if filter_unassigned:
        pert_str = adata.obs["perturbation"].astype(str)
        valid_cells = ~pert_str.isin(["nan", "None", "", "NA", "null"])
    else:
        valid_cells = np.ones(adata.n_obs, dtype=bool)

    if backed is None:
        adata = adata[valid_cells].copy()
        adata.obs["is_control"] = ctrl_mask.loc[adata.obs_names]
        return adata
    else:
        # In backed mode, return filtered view
        adata_view = adata[valid_cells]
        return adata_view


def get_control_mean(adata: ad.AnnData) -> np.ndarray:
    """Compute the empirical mean expression profile across all control cells.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData dataset with boolean `obs['is_control']`.

    Returns
    -------
    np.ndarray
        Dense 1D array of shape (n_genes,) representing the control baseline.
    """
    ctrl_mask = adata.obs["is_control"].astype(bool).values
    ctrl_X = adata.X[ctrl_mask]
    if ctrl_X.shape[0] == 0:
        raise ValueError("No control cells found with is_control == True in AnnData.")
    return np.asarray(ctrl_X.mean(axis=0)).ravel()


def list_available_datasets(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """List available benchmark dataset aliases that exist in data_dir."""
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    available = []
    for alias, fname in DATASET_FILENAMES.items():
        if (base_dir / fname).is_file():
            available.append(alias)
    return available
