import random
import numpy as np
import torch

GLOBAL_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    """Synchronize the random, NumPy, and PyTorch generators to one seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)