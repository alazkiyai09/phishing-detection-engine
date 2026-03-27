"""
PyTorch Dataset for email classification with transformers.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path

from .preprocessor import preprocess_email, truncate_text
from .tokenizer import TokenizerWrapper


class EmailDataset(Dataset):
    """
    PyTorch Dataset for email classification.

    Handles tokenization, preprocessing, and special token injection.
    """

    def __init__(
        self,
        emails: List[Dict],
        tokenizer: TokenizerWrapper,
        max_length: int = 512,
        use_special_tokens: bool = True,
        truncation_strategy: str = "head_tail"
    ):
        """
        Initialize email dataset.

        Args:
            emails: List of email dictionaries with 'text' or 'subject'/'body' and 'label'
            tokenizer: TokenizerWrapper instance
            max_length: Maximum sequence length
            use_special_tokens: Whether to use special structure tokens
            truncation_strategy: How to truncate long sequences
        """
        self.emails = emails
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_special_tokens = use_special_tokens
        self.truncation_strategy = truncation_strategy

        print(f"📊 Dataset initialized with {len(emails)} samples")
        print(f"   Max length: {max_length}")
        print(f"   Special tokens: {use_special_tokens}")
        print(f"   Truncation: {truncation_strategy}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.emails)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        email = self.emails[idx]

        # Preprocess email text
        if 'text' in email:
            # Already has combined text
            text = email['text']
        elif 'subject' in email and 'body' in email:
            # Need to combine with special tokens
            text = preprocess_email(
                email,
                use_special_tokens=self.use_special_tokens
            )
        else:
            raise ValueError("Email must have either 'text' or 'subject'/'body' fields")

        # Get label
        label = email.get('label', 0)

        # Tokenize
        encoding = self.tokenizer.tokenize(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Squeeze batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_label_distribution(self) -> Dict[int, float]:
        """
        Get distribution of labels in dataset.

        Returns:
            Dictionary mapping label to proportion
        """
        labels = [email['label'] for email in self.emails]
        total = len(labels)
        distribution = {}
        for label in set(labels):
            distribution[label] = labels.count(label) / total
        return distribution


def create_train_val_test_splits(
    data_path: str,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_state: int = 42
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create stratified train/val/test splits from CSV file.

    Args:
        data_path: Path to CSV file with email data
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_emails, val_emails, test_emails)
    """
    import numpy as np

    # Load data
    df = pd.read_csv(data_path)

    # Ensure label is integer
    if 'label' not in df.columns:
        raise ValueError(f"CSV must have 'label' column. Found: {df.columns.tolist()}")

    df['label'] = df['label'].astype(int)

    # Convert to list of dictionaries
    emails = df.to_dict('records')

    # Create local RNG for reproducibility (doesn't affect global state)
    rng = np.random.default_rng(random_state)
    rng.shuffle(emails)

    # Split by label for stratification
    emails_by_label = {0: [], 1: []}
    for email in emails:
        emails_by_label[email['label']].append(email)

    # Split each label group
    train_emails, val_emails, test_emails = [], [], []

    for label, label_emails in emails_by_label.items():
        n = len(label_emails)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        train_emails.extend(label_emails[:train_end])
        val_emails.extend(label_emails[train_end:val_end])
        test_emails.extend(label_emails[val_end:])

    # Shuffle again with local RNG
    rng.shuffle(train_emails)
    rng.shuffle(val_emails)
    rng.shuffle(test_emails)

    print(f"📊 Data splits created:")
    print(f"   Train: {len(train_emails)} samples")
    print(f"   Val: {len(val_emails)} samples")
    print(f"   Test: {len(test_emails)} samples")

    # Print class balance
    for split_name, split_data in [('Train', train_emails), ('Val', val_emails), ('Test', test_emails)]:
        phishing_count = sum(1 for e in split_data if e['label'] == 1)
        print(f"   {split_name} phishing: {phishing_count / len(split_data):.2%}")

    return train_emails, val_emails, test_emails
