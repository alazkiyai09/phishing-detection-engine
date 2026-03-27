"""
Reproducibility utilities for consistent results across runs.
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value (default: 42 from Day 2)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"🌱 Random seed set to {seed}")


def get_worker_init_fn(seed: int = 42):
    """
    Get worker initialization function for DataLoader.

    Args:
        seed: Base seed value

    Returns:
        Function to initialize each worker with different seed
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn
