#!/usr/bin/env python3
"""Create a smaller subset of the full dataset for faster training iteration."""

import pandas as pd
from pathlib import Path

def create_subset(
    input_path: str,
    output_path: str,
    n_samples: int = 5000,
    stratify: bool = True,
    seed: int = 42
):
    """
    Create a stratified subset of the dataset.

    Args:
        input_path: Path to full dataset
        output_path: Path to save subset
        n_samples: Total number of samples to include
        stratify: Whether to maintain class balance
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original dataset size: {len(df)} samples")
    print(f"Original class distribution:")
    print(df['label'].value_counts())

    if stratify:
        # Maintain class balance from original
        # Labels: 0 = legitimate, 1 = phishing
        phishing_ratio = (df['label'] == 1).mean()
        n_phishing = int(n_samples * phishing_ratio)
        n_legitimate = n_samples - n_phishing

        print(f"\nTarget samples: {n_samples} total")
        print(f"  - Phishing (label=1): {n_phishing} ({phishing_ratio:.1%})")
        print(f"  - Legitimate (label=0): {n_legitimate}")

        # Sample from each class
        phishing_df = df[df['label'] == 1].sample(
            n=n_phishing, random_state=seed
        )
        legitimate_df = df[df['label'] == 0].sample(
            n=n_legitimate, random_state=seed
        )

        subset_df = pd.concat([phishing_df, legitimate_df])
    else:
        subset_df = df.sample(n=n_samples, random_state=seed)

    # Shuffle
    subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save subset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(output_path, index=False)

    print(f"\n✅ Subset saved to {output_path}")
    print(f"Subset size: {len(subset_df)} samples")
    print(f"Subset class distribution:")
    print(subset_df['label'].value_counts())
    print(f"Subset class balance: {(subset_df['label'] == 1).mean():.1%} phishing")

if __name__ == "__main__":
    # Create subsets of different sizes
    base_path = "data/raw/phishing_emails_hf.csv"

    # Small subset for quick testing (2000 samples)
    create_subset(
        input_path=base_path,
        output_path="data/processed/phishing_emails_2k.csv",
        n_samples=2000,
        stratify=True
    )

    print("\n" + "="*70 + "\n")

    # Medium subset for reasonable training (5000 samples)
    create_subset(
        input_path=base_path,
        output_path="data/processed/phishing_emails_5k.csv",
        n_samples=5000,
        stratify=True
    )
