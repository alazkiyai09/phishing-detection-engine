#!/usr/bin/env python3
"""
Download and prepare the HuggingFace phishing email dataset.
"""
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from typing import Dict, Any


def download_phishing_dataset(
    save_path: str = "data/raw",
    dataset_name: str = "zefang-liu/phishing-email-dataset"
) -> Dict[str, Any]:
    """
    Download phishing email dataset from HuggingFace.

    Args:
        save_path: Directory to save the dataset
        dataset_name: HuggingFace dataset identifier

    Returns:
        Dictionary with dataset statistics
    """
    print(f"📥 Downloading dataset from HuggingFace: {dataset_name}")
    print(f"💾 Save path: {save_path}")

    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name)

    # Convert to pandas for easier manipulation
    train_df = dataset['train'].to_pandas()

    # Rename columns to match our schema
    # Original: 'Email Text', 'Email Type'
    # Target: 'text', 'label' (0=Safe, 1=Phishing)
    train_df.rename(columns={
        'Email Text': 'text',
        'Email Type': 'label_raw'
    }, inplace=True)

    # Convert labels to binary
    train_df['label'] = train_df['label_raw'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })

    # Drop intermediate column
    train_df.drop('label_raw', axis=1, inplace=True)

    # Drop rows with missing text
    initial_count = len(train_df)
    train_df.dropna(subset=['text'], inplace=True)
    train_df = train_df[train_df['text'].str.len() > 0]
    final_count = len(train_df)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(train_df)}")
    print(f"   Dropped samples: {initial_count - final_count}")
    print(f"   Safe emails (0): {(train_df['label'] == 0).sum()}")
    print(f"   Phishing emails (1): {(train_df['label'] == 1).sum()}")
    print(f"   Class balance: {train_df['label'].mean():.2%} phishing")

    # Save processed dataset
    output_file = Path(save_path) / "phishing_emails_hf.csv"
    train_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")

    # Show sample
    print("\n📧 Sample emails:")
    for label_val in [0, 1]:
        label_name = "Safe" if label_val == 0 else "Phishing"
        sample = train_df[train_df['label'] == label_val]['text'].iloc[0]
        print(f"\n{label_name} Email:")
        print(f"   {sample[:200]}...")

    return {
        'total_samples': len(train_df),
        'safe_emails': int((train_df['label'] == 0).sum()),
        'phishing_emails': int((train_df['label'] == 1).sum()),
        'output_file': str(output_file)
    }


if __name__ == "__main__":
    stats = download_phishing_dataset()
    print(f"\n✨ Download complete!")
    print(f"   Ready for transformer training with {stats['total_samples']} samples")
