"""
Local LLM backend using Ollama.
"""
import asyncio
import logging
import time
from typing import Optional
from .base_llm import BaseLLM, LLMResponse


logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLM):
    """
    Ollama backend for local LLM inference.

    Supports:
    - Mistral, Llama, and other Ollama models
    - Local inference (no API costs)
    - Rate limiting
    - Token counting (approximate)
    """

    def __init__(self, model_name: str = "mistral:7b", **kwargs):
        """
        Initialize Ollama backend.

        Args:
            model_name: Model name in Ollama (e.g., mistral:7b, llama2:13b)
            **kwargs: Additional parameters
                - base_url: Ollama API URL (default: http://localhost:11434)
                - timeout: Request timeout in seconds
                - num_ctx: Context window size
        """
        super().__init__(model_name, **kwargs)

        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 120)  # Longer timeout for local inference

        # Check if Ollama is available
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp package is required for OllamaBackend. "
                "Install it with: pip install aiohttp"
            )

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with self.aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Ollama.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (num_ctx, num_predict, etc.)

        Returns:
            LLMResponse
        """
        start_time = time.time()

        # Check if Ollama is available
        if not await self._check_ollama_available():
            raise ConnectionError(
                f"Ollama server not available at {self.base_url}. "
                "Make sure Ollama is running: https://ollama.ai/download"
            )

        # Prepare request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Add any additional options
        additional_options = kwargs.get("options", {})
        payload["options"].update(additional_options)

        # Make request
        try:
            async with self.aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Ollama API error: {resp.status} - {error_text}")

                    data = await resp.json()

                    latency_ms = (time.time() - start_time) * 1000

                    # Ollama returns prompt_eval_count and eval_count
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    total_tokens = prompt_tokens + completion_tokens

                    return LLMResponse(
                        content=data.get("response", ""),
                        model=self.model_name,
                        tokens_used=total_tokens,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        metadata={
                            "total_duration": data.get("total_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                            "eval_duration": data.get("eval_duration", 0)
                        }
                    )

        except self.aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.

        Note: This is a rough approximation. Ollama doesn't provide
        a tokenization API, so we estimate based on word count.

        Args:
            text: Text to count

        Returns:
            Approximate number of tokens
        """
        # Rough approximation for Mistral/Llama models
        # Average ~4 characters per token
        return int(len(text) / 4)


class OllamaBackendRateLimited(OllamaBackend):
    """
    Ollama backend with rate limiting to prevent resource exhaustion.
    """

    def __init__(self, model_name: str = "mistral:7b", concurrent_limit: int = 2, **kwargs):
        """
        Initialize with concurrency limiting.

        Args:
            model_name: Model name
            concurrent_limit: Max concurrent requests
            **kwargs: Passed to OllamaBackend
        """
        super().__init__(model_name, **kwargs)
        self.semaphore = asyncio.Semaphore(concurrent_limit)

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with concurrency limiting."""
        async with self.semaphore:
            return await super().generate(prompt, **kwargs)
