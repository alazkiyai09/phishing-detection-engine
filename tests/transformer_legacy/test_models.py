"""
Unit tests for model forward passes.
"""
import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_classifier import BERTClassifier
from src.models.roberta_classifier import RoBERTaClassifier
from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.lora_classifier import LoRABERTClassifier
from src.models.factory import create_model


class TestModelForwardPass:
    """Test model forward passes."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch."""
        return {
            'input_ids': torch.randint(0, 30522, (4, 128)),
            'attention_mask': torch.ones(4, 128),
            'labels': torch.randint(0, 2, (4,))
        }

    def test_bert_forward_pass(self, sample_batch):
        """Test BERT forward pass."""
        model = BERTClassifier(num_labels=2, dropout=0.1)
        outputs = model(**sample_batch)

        assert 'logits' in outputs
        assert 'loss' in outputs
        assert 'hidden_states' in outputs
        assert 'attentions' in outputs

        assert outputs['logits'].shape == (4, 2)
        assert outputs['loss'].item() > 0
        print("✅ test_bert_forward_pass passed")

    def test_roberta_forward_pass(self, sample_batch):
        """Test RoBERTa forward pass."""
        model = RoBERTaClassifier(num_labels=2, dropout=0.1)
        outputs = model(**sample_batch)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (4, 2)
        print("✅ test_roberta_forward_pass passed")

    def test_distilbert_forward_pass(self, sample_batch):
        """Test DistilBERT forward pass."""
        model = DistilBERTClassifier(num_labels=2, dropout=0.1)
        outputs = model(**sample_batch)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (4, 2)
        print("✅ test_distilbert_forward_pass passed")

    def test_lora_bert_forward_pass(self, sample_batch):
        """Test LoRA-BERT forward pass."""
        model = LoRABERTClassifier(num_labels=2, dropout=0.1, lora_rank=8, lora_alpha=16)
        outputs = model(**sample_batch)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (4, 2)
        print("✅ test_lora_bert_forward_pass passed")

    def test_model_factory(self):
        """Test model factory."""
        # Test BERT
        bert = create_model('bert')
        assert isinstance(bert, BERTClassifier)

        # Test RoBERTa
        roberta = create_model('roberta')
        assert isinstance(roberta, RoBERTaClassifier)

        # Test DistilBERT
        distilbert = create_model('distilbert')
        assert isinstance(distilbert, DistilBERTClassifier)

        # Test LoRA-BERT
        lora_bert = create_model('lora-bert')
        assert isinstance(lora_bert, LoRABERTClassifier)

        print("✅ test_model_factory passed")

    def test_bert_attention_extraction(self, sample_batch):
        """Test attention weight extraction."""
        model = BERTClassifier(num_labels=2, dropout=0.1)
        attention = model.get_attention_weights(
            sample_batch['input_ids'],
            sample_batch['attention_mask']
        )

        assert attention.shape[0] == 4  # batch_size
        assert attention.shape[1] == 128  # seq_len
        assert attention.shape[2] == 128  # seq_len
        print("✅ test_bert_attention_extraction passed")

    def test_bert_prediction(self, sample_batch):
        """Test prediction method."""
        model = BERTClassifier(num_labels=2, dropout=0.1)
        preds, probs = model.predict(
            sample_batch['input_ids'],
            sample_batch['attention_mask']
        )

        assert preds.shape == (4,)
        assert probs.shape == (4, 2)
        assert torch.all(preds >= 0) and torch.all(preds < 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
        print("✅ test_bert_prediction passed")

    def test_model_trainable_params(self):
        """Test that models have correct number of trainable parameters."""
        # Regular BERT
        bert = BERTClassifier(num_labels=2)
        bert_params = sum(p.numel() for p in bert.parameters() if p.requires_grad)
        assert bert_params > 100_000_000  # BERT-base has ~110M params

        # LoRA-BERT should have far fewer trainable params
        lora_bert = LoRABERTClassifier(num_labels=2, lora_rank=8, lora_alpha=16)
        lora_params = sum(p.numel() for p in lora_bert.parameters() if p.requires_grad)
        assert lora_params < 10_000_000  # LoRA should have < 10M trainable params

        print(f"   BERT params: {bert_params:,}")
        print(f"   LoRA-BERT params: {lora_params:,}")
        print(f"   Reduction: {bert_params / lora_params:.1f}x")
        print("✅ test_model_trainable_params passed")


class TestModelIO:
    """Test model saving and loading."""

    def test_bert_save_load(self, tmp_path):
        """Test saving and loading BERT model."""
        # Create model
        model = BERTClassifier(num_labels=2, dropout=0.1)

        # Save
        save_path = tmp_path / "bert_model.pt"
        model.save_pretrained(str(save_path))

        # Load
        loaded_model = BERTClassifier.from_pretrained(str(save_path))

        # Check parameters match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

        print("✅ test_bert_save_load passed")

    def test_lora_adapter_save(self, tmp_path):
        """Test saving LoRA adapters."""
        model = LoRABERTClassifier(num_labels=2, lora_rank=8, lora_alpha=16)

        # Save adapters
        adapter_path = tmp_path / "lora_adapters"
        model.save_adapters(str(adapter_path))

        # Check files exist
        assert (adapter_path / "adapter_model.bin").exists() or (adapter_path / "adapter_config.json").exists()

        print("✅ test_lora_adapter_save passed")


if __name__ == '__main__':
    # Run tests
    test_model = TestModelForwardPass()

    # Create sample batch
    sample_batch = {
        'input_ids': torch.randint(0, 30522, (4, 128)),
        'attention_mask': torch.ones(4, 128),
        'labels': torch.randint(0, 2, (4,))
    }

    test_model.test_bert_forward_pass(sample_batch)
    test_model.test_roberta_forward_pass(sample_batch)
    test_model.test_distilbert_forward_pass(sample_batch)
    test_model.test_lora_bert_forward_pass(sample_batch)
    test_model.test_model_factory()
    test_model.test_bert_attention_extraction(sample_batch)
    test_model.test_bert_prediction(sample_batch)
    test_model.test_model_trainable_params()

    print("\n✅ All model tests passed!")
