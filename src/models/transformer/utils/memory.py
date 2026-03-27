"""
GPU memory tracking utilities for monitoring training resources.
"""
import torch
import time
from typing import Dict, Optional
from contextlib import contextmanager


class GPUMemoryTracker:
    """Track GPU memory usage during training and inference."""

    def __init__(self, device: torch.device):
        """
        Initialize memory tracker.

        Args:
            device: PyTorch device (should be CUDA for GPU tracking)
        """
        self.device = device
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        self.start_memory_allocated = 0
        self.start_memory_reserved = 0
        self.tracking = False

    def start_tracking(self) -> None:
        """Start tracking memory usage."""
        if not self.device.type == 'cuda':
            print("⚠️  Not tracking memory (device is not CUDA)")
            return

        self.tracking = True
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_memory_allocated = torch.cuda.memory_allocated(self.device)
        self.start_memory_reserved = torch.cuda.memory_reserved(self.device)
        print(f"💾 Memory tracking started")
        print(f"   Initial allocated: {self.start_memory_allocated / 1024**2:.2f} MB")
        print(f"   Initial reserved: {self.start_memory_reserved / 1024**2:.2f} MB")

    def stop_tracking(self) -> Dict[str, float]:
        """
        Stop tracking and return memory statistics.

        Returns:
            Dictionary with memory statistics in MB
        """
        if not self.tracking:
            return {}

        self.peak_memory_allocated = torch.cuda.max_memory_allocated(self.device)
        self.peak_memory_reserved = torch.cuda.max_memory_reserved(self.device)
        self.tracking = False

        stats = {
            'peak_allocated_mb': self.peak_memory_allocated / 1024**2,
            'peak_reserved_mb': self.peak_memory_reserved / 1024**2,
            'allocated_increase_mb': (self.peak_memory_allocated - self.start_memory_allocated) / 1024**2,
            'reserved_increase_mb': (self.peak_memory_reserved - self.start_memory_reserved) / 1024**2,
        }

        print(f"💾 Memory tracking stopped")
        print(f"   Peak allocated: {stats['peak_allocated_mb']:.2f} MB")
        print(f"   Peak reserved: {stats['peak_reserved_mb']:.2f} MB")
        print(f"   Allocated increase: {stats['allocated_increase_mb']:.2f} MB")
        print(f"   Reserved increase: {stats['reserved_increase_mb']:.2f} MB")

        return stats

    def get_current_memory(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with current memory statistics in MB
        """
        if not self.device.type == 'cuda':
            return {}

        return {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'free_mb': (torch.cuda.get_device_properties(self.device).total_memory -
                       torch.cuda.memory_allocated(self.device)) / 1024**2,
            'total_mb': torch.cuda.get_device_properties(self.device).total_memory / 1024**2,
        }


@contextmanager
def memory_tracker_context(device: torch.device):
    """
    Context manager for tracking memory usage in a code block.

    Args:
        device: PyTorch device

    Yields:
        GPUMemoryTracker instance

    Example:
        with memory_tracker_context(device) as tracker:
            # Training code
            pass
    """
    tracker = GPUMemoryTracker(device)
    tracker.start_tracking()
    try:
        yield tracker
    finally:
        tracker.stop_tracking()


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model size statistics
    """
    param_size = 0
    trainable_param_size = 0

    for param in model.parameters():
        size = param.numel() * param.element_size()
        param_size += size
        if param.requires_grad:
            trainable_param_size += size

    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())

    total_size = param_size + buffer_size

    return {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_size_mb': total_size / 1024**2,
        'param_size_mb': param_size / 1024**2,
        'trainable_param_size_mb': trainable_param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
    }
