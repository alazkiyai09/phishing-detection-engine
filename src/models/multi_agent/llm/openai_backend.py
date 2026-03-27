"""
OpenAI GPT backend implementation for the multi-agent system.
"""
import asyncio
import logging
import time
from typing import Optional
from .base_llm import BaseLLM, LLMResponse


logger = logging.getLogger(__name__)


class OpenAIBackend(BaseLLM):
    """
    OpenAI API backend for LLM calls.

    Supports:
    - GPT-4, GPT-3.5-turbo
    - Rate limiting
    - Token counting
    - Error handling with retries
    """

    def __init__(self, model_name: str = "gpt-4", **kwargs):
        """
        Initialize OpenAI backend.

        Args:
            model_name: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            **kwargs: Additional parameters
                - api_key: OpenAI API key (or use OPENAI_API_KEY env var)
                - base_url: API base URL (default: https://api.openai.com/v1)
                - timeout: Request timeout in seconds
                - max_retries: Number of retries on failure
        """
        super().__init__(model_name, **kwargs)

        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.timeout = kwargs.get("timeout", 30)
        self.max_retries = kwargs.get("max_retries", 2)

        # Import openai here to avoid hard dependency
        try:
            import openai
            self.openai = openai
            # Initialize client
            if self.api_key:
                self.client = self.openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            else:
                # Use default API key from environment
                self.client = self.openai.AsyncOpenAI(
                    base_url=self.base_url,
                    timeout=self.timeout
                )
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIBackend. "
                "Install it with: pip install openai"
            )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from OpenAI.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        start_time = time.time()

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a security analyst specializing in phishing detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens or kwargs.get("max_tokens", 1000),
                    **{k: v for k, v in kwargs.items() if k not in ["prompt", "temperature", "max_tokens"]}
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract response data
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "attempt": attempt + 1
                    }
                )

            except self.openai.RateLimitError as e:
                last_error = e
                logger.warning(f"OpenAI rate limit hit (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    # Exponential backoff for rate limits
                    await asyncio.sleep(2 ** attempt * 2)

            except self.openai.APIError as e:
                last_error = e
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in OpenAI call (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)

        raise last_error if last_error else Exception("OpenAI API call failed")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        try:
            import tiktoken

            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to cl100k_base (GPT-4 encoding)
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough approximation
            return int(len(text.split()) / 0.75)


class OpenAIBackendRateLimited(OpenAIBackend):
    """
    OpenAI backend with built-in rate limiting.
    """

    def __init__(self, model_name: str = "gpt-4", requests_per_minute: int = 60, **kwargs):
        """
        Initialize with rate limiting.

        Args:
            model_name: Model name
            requests_per_minute: Max requests per minute
            **kwargs: Passed to OpenAIBackend
        """
        super().__init__(model_name, **kwargs)
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with rate limiting."""
        async with self._lock:
            # Clean old requests (older than 1 minute)
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]

            # Check rate limit
            if len(self.request_times) >= self.requests_per_minute:
                # Wait until we can make a request
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Clean again
                    self.request_times = []

            # Record this request
            self.request_times.append(now)

            # Make the actual call
            return await super().generate(prompt, **kwargs)
