"""
Learning rate scheduler with linear warmup and decay.
"""
import math
from typing import Optional
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly
    after linearly increasing during a warmup period.

    Based on HuggingFace implementation.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Warmup phase: increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decay phase: decrease from 1 to 0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LinearWarmupScheduler:
    """
    Learning rate scheduler with linear warmup and decay.
    Matches the scheduler used in BERT pretraining.
    """

    def __init__(
        self,
        optimizer,
        warmup_ratio: float,
        total_steps: int
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_ratio: Proportion of steps for warmup (e.g., 0.1 = 10%)
            total_steps: Total number of training steps
        """
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.total_steps = total_steps
        self.optimizer = optimizer

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        print(f"📈 Learning rate scheduler initialized")
        print(f"   Warmup steps: {self.warmup_steps} ({warmup_ratio:.0%})")
        print(f"   Total steps: {total_steps}")

    def step(self) -> None:
        """Take a scheduler step."""
        self.scheduler.step()

    def get_last_lr(self) -> list:
        """Get last learning rate."""
        return self.scheduler.get_last_lr()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.get_last_lr()[0]

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)


def calculate_total_steps(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1
) -> int:
    """
    Calculate total number of training steps.

    Args:
        num_samples: Number of training samples
        batch_size: Batch size per device
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Total training steps
    """
    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    return steps_per_epoch * num_epochs
