"""Ollama backend implementation for local LLM support."""

import asyncio
import json
import logging
import time
from typing import Optional

from .base_llm import BaseLLM, LLMResponse


logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLM):
    """
    Ollama backend for running local LLMs (e.g., Mistral-7B, Llama-3).

    Supports:
    - Local model execution via Ollama API
    - No API costs
    - Privacy (data doesn't leave your machine)
    - Common models: mistral, llama3, codellama, etc.
    """

    def __init__(self, model_name: str = "mistral:7b", **kwargs):
        """Initialize Ollama backend.

        Args:
            model_name: Model to use (e.g., mistral:7b, llama3:8b)
            **kwargs: Additional parameters
                - host: Ollama host URL (default: http://localhost:11434)
                - timeout: Request timeout in seconds
                - num_ctx: Context window size
        """
        super().__init__(model_name, **kwargs)

        self.host = kwargs.get("host", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 120)
        self.num_ctx = kwargs.get("num_ctx", 4096)

        # Import aiohttp for HTTP requests
        try:
            import aiohttp

            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp package is required for OllamaBackend. "
                "Install it with: pip install aiohttp"
            )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from Ollama.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        start_time = time.time()

        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Add any additional options
        if "options" in kwargs:
            payload["options"].update(kwargs["options"])

        try:
            async with self.aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Ollama API error: {response.status} - {error_text}"
                        )

                    data = await response.json()

                    latency_ms = (time.time() - start_time) * 1000

                    # Extract response data
                    content = data.get("response", "")
                    prompt_tokens = self.count_tokens(prompt)
                    completion_tokens = self.count_tokens(content)
                    total_tokens = prompt_tokens + completion_tokens

                    return LLMResponse(
                        content=content,
                        model=self.model_name,
                        tokens_used=total_tokens,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        metadata={
                            "eval_duration": data.get("eval_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "total_duration": data.get("total_duration", 0),
                        },
                    )

        except self.aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.host}: {e}")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Note: This is a rough approximation. Ollama models use different tokenizers.
        For accurate counts, you'd need to use the specific model's tokenizer.

        Args:
            text: Text to count

        Returns:
            Approximate number of tokens
        """
        # Rough approximation: ~0.75 words per token
        return int(len(text.split()) / 0.75)

    async def check_connection(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if connected, False otherwise
        """
        try:
            url = f"{self.host}/api/tags"
            async with self.aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama: {e}")
            return False

    async def list_models(self) -> list:
        """List available models in Ollama.

        Returns:
            List of model names
        """
        url = f"{self.host}/api/tags"

        try:
            async with self.aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")

        return []
