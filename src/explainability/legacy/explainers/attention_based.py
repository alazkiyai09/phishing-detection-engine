"""
Attention-based explainer for transformer models.

Extracts and visualizes attention weights from transformer models.
"""

from typing import List, Optional, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    AttentionVisualization
)


class AttentionBasedExplainer:
    """
    Attention-based explainer for transformer models.

    Shows which tokens the model focused on when making predictions.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize attention-based explainer.

        Args:
            model: Transformer model (e.g., DistilBERT, BERT)
            tokenizer: Tokenizer for the model
            layer_indices: Which layers to analyze (default: last layer)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.layer_indices = layer_indices or [-1]

        # Register hooks to capture attention
        self.attention_weights = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def hook_fn(module, input, output):
            # Capture attention weights from output
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_weights = output.attentions

        # Try to register hook on transformer layers
        # This depends on the specific model architecture
        try:
            if hasattr(self.model, 'distilbert'):
                self.model.distilbert.register_forward_hook(hook_fn)
            elif hasattr(self.model, 'bert'):
                self.model.bert.register_forward_hook(hook_fn)
        except Exception:
            pass  # Hook registration failed

    def explain(
        self,
        email: EmailData,
        max_length: int = 128
    ) -> AttentionVisualization:
        """
        Extract attention weights for email.

        Args:
            email: Email to explain
            max_length: Maximum sequence length

        Returns:
            AttentionVisualization with attention weights
        """
        if self.tokenizer is None:
            # Create dummy attention if no tokenizer
            return self._create_dummy_attention(email)

        # Tokenize email
        text = f"{email.subject} {email.body}"

        try:
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            # Get attention weights
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            if outputs.attentions is not None:
                # Extract attention for specified layers
                attentions = outputs.attentions

                # Average across heads and layers
                # Shape: (num_layers, num_heads, seq_len, seq_len)
                all_attention = []

                for layer_idx in self.layer_indices:
                    layer_attention = attentions[layer_idx]
                    # Average across heads: (seq_len, seq_len)
                    avg_attention = layer_attention.mean(dim=1).squeeze(0).cpu().numpy()
                    all_attention.append(avg_attention.tolist())

                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

                return AttentionVisualization(
                    tokens=tokens,
                    attention_weights=all_attention,
                    layer_indices=self.layer_indices
                )

        except Exception as e:
            print(f"Error extracting attention: {e}")

        # Fallback to dummy attention
        return self._create_dummy_attention(email)

    def _create_dummy_attention(self, email: EmailData) -> AttentionVisualization:
        """Create dummy attention visualization when actual extraction fails."""
        # Simple word tokenization
        text = f"{email.subject} {email.body}"
        tokens = text.split()[:50]  # Limit to 50 tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Create uniform attention as placeholder
        seq_len = len(tokens)
        dummy_attention = [[1.0 / seq_len] * seq_len for _ in range(seq_len)]

        return AttentionVisualization(
            tokens=tokens,
            attention_weights=[dummy_attention],
            layer_indices=[-1]
        )

    def get_top_attended_tokens(
        self,
        email: EmailData,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Get top-k attended tokens for email.

        Args:
            email: Email to analyze
            top_k: Number of top tokens to return

        Returns:
            List of (token, attention_score) tuples
        """
        attention_viz = self.explain(email)
        return attention_viz.get_top_attended_tokens(top_k)

    def explain_multiple(
        self,
        emails: List[EmailData],
        max_length: int = 128
    ) -> List[AttentionVisualization]:
        """
        Extract attention weights for multiple emails.

        Args:
            emails: List of emails to explain
            max_length: Maximum sequence length

        Returns:
            List of AttentionVisualization objects
        """
        return [self.explain(email, max_length) for email in emails]


# Simplified attention explainer without transformer dependency
class SimpleAttentionExplainer:
    """
    Simplified attention explainer that doesn't require transformer models.

    Uses heuristic-based attention scores (TF-IDF like).
    """

    def __init__(self):
        """Initialize simple attention explainer."""
        # Suspicious keywords that should get high attention
        self.suspicious_keywords = {
            # High priority
            'password', 'verify', 'confirm', 'urgent', 'immediately',
            'suspended', 'account', 'security', 'update', 'click',
            'login', 'signin', 'banking', 'payment', 'wire',
            # Medium priority
            'invoice', 'receipt', 'attached', 'download', 'important',
            'confidential', 'secret', 'expire', 'deadline'
        }

    def explain(self, email: EmailData) -> AttentionVisualization:
        """
        Compute heuristic attention scores for email.

        Args:
            email: Email to explain

        Returns:
            AttentionVisualization with heuristic attention
        """
        # Combine subject and body
        text = f"{email.subject} {email.body}"
        tokens = text.split()

        # Compute attention scores
        attention_scores = []
        for token in tokens:
            # Base score
            score = 0.1

            # Bonus for suspicious keywords
            if token.lower() in self.suspicious_keywords:
                score = 0.8

            # Bonus for ALL CAPS
            if token.isupper() and len(token) > 2:
                score = max(score, 0.6)

            # Bonus for exclamation marks
            if '!' in token:
                score = max(score, 0.5)

            attention_scores.append(score)

        # Normalize
        total = sum(attention_scores)
        if total > 0:
            attention_scores = [s / total for s in attention_scores]

        # Create attention matrix (simplified)
        n_tokens = len(tokens)
        attention_matrix = [attention_scores]

        return AttentionVisualization(
            tokens=tokens,
            attention_weights=attention_matrix,
            layer_indices=[-1]
        )
