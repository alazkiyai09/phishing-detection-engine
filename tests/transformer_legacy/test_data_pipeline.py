"""
Unit tests for data pipeline.
"""
import pytest
import torch
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import EmailDataset
from src.data.tokenizer import TokenizerWrapper
from src.data.preprocessor import preprocess_email, truncate_text, clean_text


class TestDataPreprocessing:
    """Test email preprocessing functions."""

    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Hello    world\n\n  Test  "
        cleaned = clean_text(text)
        assert cleaned == "Hello world Test"
        print("✅ test_clean_text passed")

    def test_preprocess_email_with_dict(self):
        """Test preprocessing email dict."""
        email = {
            'subject': 'URGENT: Verify Account',
            'body': 'Click here to verify',
            'sender': 'fake@phishing.com',
            'url': 'http://bad-site.com'
        }

        processed = preprocess_email(email, use_special_tokens=True)

        assert '[SUBJECT]' in processed
        assert '[BODY]' in processed
        assert '[URL]' in processed
        assert '[SENDER]' in processed
        print("✅ test_preprocess_email_with_dict passed")

    def test_preprocess_email_without_special_tokens(self):
        """Test preprocessing without special tokens."""
        email = {
            'subject': 'Test Subject',
            'body': 'Test Body'
        }

        processed = preprocess_email(email, use_special_tokens=False)

        assert '[SUBJECT]' not in processed
        assert 'Test Subject' in processed
        assert 'Test Body' in processed
        print("✅ test_preprocess_email_without_special_tokens passed")

    def test_truncate_text_head_only(self):
        """Test head-only truncation."""
        tokens = list(range(100))
        truncated = truncate_text(tokens, max_length=50, strategy='head_only')
        assert len(truncated) == 50
        assert truncated == tokens[:50]
        print("✅ test_truncate_text_head_only passed")

    def test_truncate_text_tail_only(self):
        """Test tail-only truncation."""
        tokens = list(range(100))
        truncated = truncate_text(tokens, max_length=50, strategy='tail_only')
        assert len(truncated) == 50
        assert truncated == tokens[-50:]
        print("✅ test_truncate_text_tail_only passed")

    def test_truncate_text_head_tail(self):
        """Test head+tail truncation."""
        tokens = list(range(100))
        truncated = truncate_text(tokens, max_length=53, strategy='head_tail')
        assert len(truncated) == 51  # 25 + 1 + 25
        assert '[TRUNCATED]' in truncated
        print("✅ test_truncate_text_head_tail passed")


class TestTokenizer:
    """Test tokenizer wrapper."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tiny tokenizer for testing."""
        # Use distilbert which is faster to load
        return TokenizerWrapper('distilbert-base-uncased', cache_dir='/tmp/test_cache')

    def test_tokenize(self, tokenizer):
        """Test basic tokenization."""
        text = "This is a test email"
        result = tokenizer.tokenize(text, max_length=20)

        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert result['input_ids'].shape[0] == 1
        assert result['input_ids'].shape[1] <= 20
        print("✅ test_tokenize passed")

    def test_decode(self, tokenizer):
        """Test decoding."""
        token_ids = [101, 2023, 2003, 1037, 3231, 102]  # [CLS] This is a test [SEP]
        decoded = tokenizer.decode(token_ids)

        assert 'test' in decoded.lower()
        print("✅ test_decode passed")


class TestEmailDataset:
    """Test EmailDataset."""

    @pytest.fixture
    def sample_emails(self):
        """Create sample emails."""
        return [
            {
                'text': 'URGENT: Click here now',
                'label': 1
            },
            {
                'text': 'Hello, how are you?',
                'label': 0
            },
            {
                'subject': 'Meeting tomorrow',
                'body': 'Lets meet at 2pm',
                'label': 0
            }
        ]

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer."""
        return TokenizerWrapper('distilbert-base-uncased', cache_dir='/tmp/test_cache')

    def test_dataset_length(self, sample_emails, tokenizer):
        """Test dataset length."""
        dataset = EmailDataset(sample_emails, tokenizer, max_length=64)
        assert len(dataset) == 3
        print("✅ test_dataset_length passed")

    def test_dataset_getitem(self, sample_emails, tokenizer):
        """Test getting items from dataset."""
        dataset = EmailDataset(sample_emails, tokenizer, max_length=64)
        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
        assert item['input_ids'].shape[0] == 64
        assert item['labels'].item() == 1  # First sample is phishing
        print("✅ test_dataset_getitem passed")

    def test_dataset_with_subject_body(self, sample_emails, tokenizer):
        """Test dataset with separate subject/body."""
        dataset = EmailDataset(sample_emails, tokenizer, max_length=64, use_special_tokens=True)

        # Third item has subject/body
        item = dataset[2]
        assert item['labels'].item() == 0  # Legitimate
        print("✅ test_dataset_with_subject_body passed")


if __name__ == '__main__':
    # Run tests
    test_preprocessing = TestDataPreprocessing()
    test_preprocessing.test_clean_text()
    test_preprocessing.test_preprocess_email_with_dict()
    test_preprocessing.test_preprocess_email_without_special_tokens()
    test_preprocessing.test_truncate_text_head_only()
    test_preprocessing.test_truncate_text_tail_only()
    test_preprocessing.test_truncate_text_head_tail()

    print("\n✅ All data pipeline tests passed!")
