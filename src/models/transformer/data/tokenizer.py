"""
Tokenizer wrapper with caching support for HuggingFace models.
"""
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """
    Wrapper around HuggingFace tokenizer with caching support.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "data/cache",
        use_cache: bool = True
    ):
        """
        Initialize tokenizer wrapper.

        Args:
            model_name: HuggingFace model name
            cache_dir: Directory for caching
            use_cache: Whether to use caching
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir / "hf_cache")
        )

        # Define and add special tokens if needed
        special_tokens_list = ["[SUBJECT]", "[BODY]", "[URL]", "[SENDER]", "[TRUNCATED]"]
        special_tokens = {
            "additional_special_tokens": special_tokens_list
        }

        # Get existing special tokens before adding
        existing_added = self.tokenizer.added_tokens_encoder.copy()

        # Add special tokens
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # Validate special tokens were added successfully
        added_tokens = self.tokenizer.added_tokens_encoder
        missing_tokens = []
        for token in special_tokens_list:
            if token not in added_tokens:
                missing_tokens.append(token)

        if missing_tokens:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Special tokens may not have been added correctly: {missing_tokens}")
            logger.warning(f"Added tokens encoder: {added_tokens}")

        print(f"🔤 Tokenizer loaded: {model_name}")
        print(f"   Vocab size: {len(self.tokenizer)} (added {num_added} special tokens)")
        print(f"   Max model length: {self.tokenizer.model_max_length}")

        # Verify special tokens are single tokens
        for token in special_tokens_list:
            if token in added_tokens:
                token_id = added_tokens[token]
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if decoded != token:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Special token '{token}' (id={token_id}) decodes to '{decoded}', not '{token}'. "
                        f"This may indicate the tokenizer doesn't support this token properly."
                    )

    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: str = "pt"
    ) -> Dict[str, any]:
        """
        Tokenize text(s).

        Args:
            texts: Text or list of texts to tokenize
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences
            return_tensors: Format to return ('pt', 'np', or None)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_cache_key(self, text: str, max_length: int) -> str:
        """
        Generate cache key for tokenized text.

        Args:
            text: Input text
            max_length: Max length used for tokenization

        Returns:
            Cache key string
        """
        content = f"{self.model_name}_{max_length}_{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def tokenize_with_cache(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, List[int]]:
        """
        Tokenize text with caching support.

        Args:
            text: Text to tokenize
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        cache_key = self.get_cache_key(text, max_length)
        # Use .npz instead of .pkl for safer loading
        cache_file = self.cache_dir / f"tokens_{cache_key}.npz"

        # Try loading from cache using numpy (safer than pickle)
        if self.use_cache and cache_file.exists():
            try:
                import numpy as np
                cached_data = np.load(cache_file, allow_pickle=False)
                # Convert back to expected format
                return {
                    'input_ids': cached_data['input_ids'].tolist(),
                    'attention_mask': cached_data['attention_mask'].tolist()
                }
            except (OSError, ValueError, EOFError) as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}. Retokenizing...")

        # Tokenize
        tokenized = self.tokenize(text, max_length=max_length)

        # Save to cache using numpy
        if self.use_cache:
            try:
                import numpy as np
                # Convert tensors to numpy arrays for safe storage
                input_ids = tokenized['input_ids'].cpu().numpy() if hasattr(tokenized['input_ids'], 'cpu') else tokenized['input_ids']
                attention_mask = tokenized['attention_mask'].cpu().numpy() if hasattr(tokenized['attention_mask'], 'cpu') else tokenized['attention_mask']
                np.savez_compressed(cache_file, input_ids=input_ids, attention_mask=attention_mask)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")

        return tokenized

    def clear_cache(self) -> None:
        """Clear all cached tokenizations."""
        # Clear both old .pkl and new .npz cache files
        cache_files = list(self.cache_dir.glob("tokens_*.pkl")) + list(self.cache_dir.glob("tokens_*.npz"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        print(f"🗑️  Cleared {len(cache_files)} cached tokenizations")

    def save_pretrained(self, path: str) -> None:
        """
        Save tokenizer to directory.

        Args:
            path: Directory path
        """
        self.tokenizer.save_pretrained(path)
        print(f"💾 Tokenizer saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'TokenizerWrapper':
        """
        Load tokenizer wrapper from directory.

        Args:
            path: Directory path
            **kwargs: Additional arguments

        Returns:
            TokenizerWrapper instance
        """
        tokenizer = AutoTokenizer.from_pretrained(path)
        wrapper = cls.__new__(cls)
        wrapper.tokenizer = tokenizer
        wrapper.model_name = path
        wrapper.cache_dir = Path(kwargs.get('cache_dir', 'data/cache'))
        wrapper.use_cache = kwargs.get('use_cache', True)
        return wrapper
