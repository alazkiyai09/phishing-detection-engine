"""
Trainer class for transformer-based phishing detection.
Handles training loop, validation, early stopping, FP16, and gradient accumulation.
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .metrics import compute_all_metrics, compute_auprc
from .scheduler import LinearWarmupScheduler, calculate_total_steps
from ..utils.memory import GPUMemoryTracker
from ..utils.config import TrainingConfig


class Trainer:
    """
    Trainer for transformer-based email classification.

    Features:
    - FP16 mixed precision training
    - Gradient accumulation
    - Linear warmup + decay LR scheduling
    - Early stopping on validation AUPRC
    - Weights & Biases logging
    - Checkpoint saving
    - GPU memory tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        config: TrainingConfig,
        tokenizer=None
    ):
        """
        Initialize trainer.

        Args:
            model: Transformer model
            train_dataset: Training data loader
            val_dataset: Validation data loader
            config: Training configuration
            tokenizer: Tokenizer (for saving with checkpoints)
        """
        self.model = model
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.config = config
        self.tokenizer = tokenizer

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        total_steps = calculate_total_steps(
            num_samples=len(train_dataset.dataset),
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )

        self.scheduler = LinearWarmupScheduler(
            self.optimizer,
            warmup_ratio=config.warmup_ratio,
            total_steps=total_steps
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

        # Early stopping
        self.best_metric = 0.0
        self.patience_counter = 0
        self.early_stop = False

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auprc': [],
            'val_auroc': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # GPU memory tracker
        self.memory_tracker = GPUMemoryTracker(self.device)

        # W&B logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )

        print(f"\n🚀 Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   FP16: {config.fp16}")
        print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"   Effective batch size: {config.effective_batch_size()}")
        print(f"   Total steps: {total_steps}")

    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'='*60}\n")

        self.memory_tracker.start_tracking()

        for epoch in range(self.config.num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auprc'].append(val_metrics['auprc'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(self.scheduler.get_lr())

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Check early stopping
            if self._check_early_stopping(val_metrics):
                print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0 or val_metrics['auprc'] > self.best_metric:
                self._save_checkpoint(epoch, val_metrics)

        # Get memory stats
        memory_stats = self.memory_tracker.stop_tracking()

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation AUPRC: {self.best_metric:.4f}")
        print(f"{'='*60}\n")

        return self.history

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False
        )

        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}"
            })

        avg_loss = total_loss / num_batches

        return {
            'loss': avg_loss,
            'learning_rate': self.scheduler.get_lr()
        }

    @torch.no_grad()
    def validate(self, financial_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            financial_mask: Optional boolean mask for financial phishing subset [n_samples]

        Returns:
            Dictionary with validation metrics
        """
        import logging
        logger = logging.getLogger(__name__)

        self.model.eval()

        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()

        # Compute metrics
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = compute_all_metrics(all_probs, all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)

        # Financial subset analysis (if mask provided)
        if financial_mask is not None:
            if len(financial_mask) != len(all_labels):
                logger.warning(
                    f"Financial mask length ({len(financial_mask)}) != "
                    f"validation samples ({len(all_labels)}). Skipping financial analysis."
                )
            else:
                financial_count = financial_mask.sum()
                if financial_count > 0:
                    logger.info(f"Computing metrics for {financial_count} financial phishing samples")

                    # Extract financial subset
                    financial_labels = all_labels[financial_mask]
                    financial_preds = all_preds[financial_mask]
                    financial_probs = all_probs[financial_mask]

                    # Compute financial-specific metrics
                    financial_metrics = compute_all_metrics(financial_probs, financial_preds, financial_labels)

                    # Add with 'financial_' prefix
                    metrics['financial_recall'] = financial_metrics['recall']
                    metrics['financial_fpr'] = financial_metrics['fpr']
                    metrics['financial_f1'] = financial_metrics['f1']
                    metrics['financial_auprc'] = financial_metrics['auprc']
                    metrics['financial_count'] = financial_count

                    # Log financial metrics
                    print(f"   Financial ({financial_count} samples): "
                          f"Recall={metrics['financial_recall']:.4f}, "
                          f"FPR={metrics['financial_fpr']:.4f}, "
                          f"F1={metrics['financial_f1']:.4f}")

                    # Check financial sector requirements
                    meets_recall = metrics['financial_recall'] >= 0.95
                    meets_fpr = metrics['financial_fpr'] <= 0.01
                    metrics['financial_requirements_met'] = meets_recall and meets_fpr

                    if not metrics['financial_requirements_met']:
                        logger.warning(
                            f"Financial requirements NOT met: "
                            f"Recall={metrics['financial_recall']:.2%} (need >=95%), "
                            f"FPR={metrics['financial_fpr']:.2%} (need <=1%)"
                        )
                else:
                    logger.info("No financial phishing samples in validation set")

        print(f"   AUPRC: {metrics['auprc']:.4f} | AUROC: {metrics['auroc']:.4f} | "
              f"Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping should trigger.

        Args:
            val_metrics: Validation metrics

        Returns:
            True if early stopping should trigger
        """
        metric = val_metrics[self.config.early_stopping_metric]

        if metric > self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
            return False

        self.patience_counter += 1

        if self.patience_counter >= self.config.early_stopping_patience:
            return True

        return False

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics to console and W&B."""
        # Console
        print(f"\n   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val AUPRC: {val_metrics['auprc']:.4f}")
        print(f"   Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   LR: {train_metrics['learning_rate']:.2e}")

        # W&B
        if self.config.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_auprc': val_metrics['auprc'],
                'val_auroc': val_metrics['auroc'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': train_metrics['learning_rate']
            }
            wandb.log(log_dict)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config.__dict__
        }, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_metrics['auprc'] == self.best_metric:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'val_metrics': val_metrics,
                'config': self.config.__dict__
            }, best_path)
            print(f"🏆 Best model saved: {best_path}")

            # Save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(checkpoint_dir))
