#!/usr/bin/env python3
"""
Main training script for transformer-based phishing detection.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import TrainingConfig, MODEL_CONFIGS
from src.utils.seed import set_seed
from src.data.tokenizer import TokenizerWrapper
from src.data.dataset import EmailDataset, create_train_val_test_splits
from src.models.factory import create_model
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train transformer phishing detector')

    parser.add_argument(
        '--model',
        type=str,
        choices=['bert', 'roberta', 'distilbert', 'lora-bert'],
        default='bert',
        help='Model type to train'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (overrides model defaults)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/phishing_emails_hf.csv',
        help='Path to training data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for checkpoints'
    )

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"🚀 Transformer Phishing Detection Training")
    print(f"{'='*70}\n")

    # Set random seed
    set_seed(args.seed)

    # Load or create config
    if args.config:
        print(f"📋 Loading config from: {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"📋 Using default config for {args.model}")
        model_config = MODEL_CONFIGS[args.model]
        config = model_config.to_training_config(TrainingConfig())

    # Override config with command line args
    if args.output:
        config.output_dir = args.output
    if args.no_wandb:
        config.use_wandb = False
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.data:
        config.data_path = args.data
    config.seed = args.seed

    print(f"\n⚙️  Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   LoRA: {config.use_lora}")
    print(f"   Learning rate: {config.learning_rate:.2e}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   FP16: {config.fp16}")
    print(f"   Device: {config.device}")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_save_path = Path(config.output_dir) / 'config.yaml'
    config.to_yaml(str(config_save_path))

    # Load data
    print(f"\n📂 Loading data from: {config.data_path}")
    train_emails, val_emails, test_emails = create_train_val_test_splits(
        data_path=config.data_path,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        random_state=config.seed
    )

    # Create tokenizer
    print(f"\n🔤 Creating tokenizer...")
    tokenizer = TokenizerWrapper(
        model_name=config.model_name,
        cache_dir=config.cache_dir
    )

    # Create datasets
    print(f"\n📊 Creating datasets...")
    train_dataset = EmailDataset(
        emails=train_emails,
        tokenizer=tokenizer,
        max_length=config.max_length,
        use_special_tokens=config.use_special_tokens,
        truncation_strategy=config.truncation_strategy
    )

    val_dataset = EmailDataset(
        emails=val_emails,
        tokenizer=tokenizer,
        max_length=config.max_length,
        use_special_tokens=config.use_special_tokens,
        truncation_strategy=config.truncation_strategy
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )

    # Create model
    print(f"\n🤖 Creating model: {args.model}")
    model = create_model(
        model_type=args.model,
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout=config.dropout,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha
    )

    # Create trainer
    print(f"\n🏋️  Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        config=config,
        tokenizer=tokenizer
    )

    # Train
    print(f"\n{'='*70}")
    print(f"Starting training...")
    print(f"{'='*70}\n")

    history = trainer.train()

    # Save final model
    print(f"\n💾 Saving final model...")
    final_model_path = Path(config.output_dir) / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'history': history
    }, final_model_path)

    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"   Model saved to: {final_model_path}")
    print(f"   Best validation AUPRC: {trainer.best_metric:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
