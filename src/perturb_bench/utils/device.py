"""Hardware device resolution (MPS for Apple Silicon, CUDA for NVIDIA, CPU)."""
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Detect and return the optimal compute device.

    Parameters
    ----------
    prefer_mps : bool, default=True
        Whether to select Apple Silicon Metal Performance Shaders (MPS)
        if available.

    Returns
    -------
    torch.device
        Detected PyTorch device ('mps', 'cuda', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
