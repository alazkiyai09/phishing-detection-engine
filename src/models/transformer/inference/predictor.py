"""
Prediction wrapper for transformer-based phishing detection.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

from ..models.base import BaseTransformerClassifier
from ..data.tokenizer import TokenizerWrapper
from ..data.preprocessor import preprocess_email


class Predictor:
    """
    High-level prediction interface for phishing detection.

    Handles:
    - Single email prediction
    - Batch prediction
    - Attention extraction
    - Confidence calibration
    """

    def __init__(
        self,
        model: BaseTransformerClassifier,
        tokenizer: TokenizerWrapper,
        device: str = "cuda"
    ):
        """
        Initialize predictor.

        Args:
            model: Trained transformer model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"🔮 Predictor initialized on {self.device}")

    @torch.no_grad()
    def predict(
        self,
        email: Union[str, Dict[str, str]],
        return_attention: bool = False,
        use_special_tokens: bool = True
    ) -> Dict[str, any]:
        """
        Predict phishing probability for a single email.

        Args:
            email: Email text or dict with subject/body
            return_attention: Whether to return attention weights
            use_special_tokens: Whether to use special structure tokens

        Returns:
            Dictionary with prediction results
        """
        # Preprocess email
        if isinstance(email, str):
            text = email
        elif isinstance(email, dict):
            text = preprocess_email(email, use_special_tokens=use_special_tokens)
        else:
            raise ValueError("email must be str or dict")

        # Tokenize
        encoding = self.tokenizer.tokenize(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()

        result = {
            'label': int(pred),
            'label_text': 'Phishing' if pred == 1 else 'Safe',
            'confidence': confidence,
            'probabilities': {
                'safe': probs[0, 0].item(),
                'phishing': probs[0, 1].item()
            }
        }

        # Add attention if requested
        if return_attention:
            attention = self.model.get_attention_weights(input_ids, attention_mask)
            result['attention'] = attention.cpu().numpy()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        emails: List[Union[str, Dict[str, str]]],
        batch_size: int = 32,
        use_special_tokens: bool = True
    ) -> List[Dict[str, any]]:
        """
        Predict phishing probability for multiple emails.

        Args:
            emails: List of email texts or dicts
            batch_size: Batch size for inference
            use_special_tokens: Whether to use special structure tokens

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(emails), batch_size):
            batch_emails = emails[i:i + batch_size]

            # Preprocess
            texts = []
            for email in batch_emails:
                if isinstance(email, str):
                    texts.append(email)
                else:
                    texts.append(preprocess_email(email, use_special_tokens=use_special_tokens))

            # Tokenize batch
            encoding = self.tokenizer.tokenize(
                texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # Collect results
            for j, pred in enumerate(preds):
                results.append({
                    'label': int(pred.item()),
                    'label_text': 'Phishing' if pred.item() == 1 else 'Safe',
                    'confidence': probs[j, pred].item(),
                    'probabilities': {
                        'safe': probs[j, 0].item(),
                        'phishing': probs[j, 1].item()
                    }
                })

        return results

    def get_attention_for_email(
        self,
        email: Union[str, Dict[str, str]],
        layer: int = -1,
        head: int = 0
    ) -> Dict[str, any]:
        """
        Extract attention weights for visualization.

        Args:
            email: Email text or dict
            layer: Which layer to extract
            head: Which head to extract

        Returns:
            Dictionary with attention data and tokens
        """
        # Preprocess
        if isinstance(email, str):
            text = email
        else:
            text = preprocess_email(email, use_special_tokens=True)

        # Tokenize
        encoding = self.tokenizer.tokenize(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get attention
        attention = self.model.get_attention_weights(
            input_ids,
            attention_mask,
            layer=layer,
            head=head
        )

        # Get tokens
        tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().numpy()
        )

        # Remove padding
        real_len = attention_mask[0].sum().item()
        tokens = tokens[:real_len]
        attention = attention[:, :real_len, :real_len]

        return {
            'attention': attention.cpu().numpy(),
            'tokens': tokens,
            'layer': layer,
            'head': head
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda"
    ) -> 'Predictor':
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on

        Returns:
            Predictor instance
        """
        from ..models.factory import create_model
        from ..utils.config import TrainingConfig

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Reconstruct config
        if 'config' in checkpoint:
            config = TrainingConfig(**checkpoint['config'])
        else:
            config = TrainingConfig()  # Use defaults

        # Create model
        model_type = config.model_name.split('-')[0]  # Extract 'bert' from 'bert-base-uncased'
        model = create_model(
            model_type=model_type,
            model_name=config.model_name,
            num_labels=config.num_labels,
            dropout=config.dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])

        # Create tokenizer
        tokenizer = TokenizerWrapper(config.model_name)

        return cls(model, tokenizer, device)
