"""
GLM (Zhipu AI) backend for multi-agent system.

This module provides an OpenAI-compatible interface to GLM models.
"""
import os
import asyncio
from typing import Dict, Any, Optional, List
import aiohttp
import logging

from .base_llm import BaseLLM
from ..utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class GLMBackend(BaseLLM):
    """
    GLM (Zhipu AI) backend implementation.

    Uses Zhipu AI's API which is compatible with OpenAI's format.
    API Documentation: https://open.bigmodel.cn/dev/api
    """

    # Available GLM models
    MODELS = {
        "glm-4-flash": {
            "context_length": 128000,
            "input_price": 0.1,  # RMB per 1M tokens
            "output_price": 0.1,
            "description": "Fast, cost-effective model"
        },
        "glm-4-air": {
            "context_length": 128000,
            "input_price": 1.0,
            "output_price": 1.0,
            "description": "Balanced performance and speed"
        },
        "glm-4-plus": {
            "context_length": 128000,
            "input_price": 5.0,
            "output_price": 5.0,
            "description": "Highest capability model"
        }
    }

    def __init__(
        self,
        model_name: str = "glm-4-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize GLM backend.

        Args:
            model_name: GLM model to use
            api_key: GLM API key (uses GLM_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            cost_tracker: Optional cost tracker instance
        """
        super().__init__(model_name, temperature, max_tokens)

        self.api_key = api_key or os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError("GLM_API_KEY environment variable must be set")

        self.cost_tracker = cost_tracker or CostTracker()

        # Model pricing info
        if model_name not in self.MODELS:
            logger.warning(f"Unknown model {model_name}, using default pricing")
        model_info = self.MODELS.get(model_name, self.MODELS["glm-4-flash"])

        self.input_price = model_info["input_price"]
        self.output_price = model_info["output_price"]

        # API endpoint
        self.api_base = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

        logger.info(f"Initialized GLM backend with model {model_name}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        retries: int = 3
    ) -> str:
        """
        Generate response using GLM API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: Optional response format (e.g., {"type": "json_object"})
            retries: Number of retries on failure

        Returns:
            Generated text response
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.7,
            "stream": False
        }

        # Add response format if specified
        if response_format:
            request_body["response_format"] = response_format

        # Retry logic
        last_error = None
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_base,
                        headers=headers,
                        json=request_body,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:

                        if response.status != 200:
                            error_text = await response.text()
                            logger.warning(
                                f"GLM API error (attempt {attempt + 1}/{retries}): "
                                f"{response.status} - {error_text}"
                            )
                            last_error = error_text

                            # Don't retry on authentication errors
                            if response.status == 401:
                                raise ValueError("Invalid GLM API key")

                            # Retry on rate limits or server errors
                            if response.status in [429, 500, 502, 503, 504]:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue

                            raise RuntimeError(f"GLM API error: {error_text}")

                        data = await response.json()

                        # Extract response
                        content = data["choices"][0]["message"]["content"]

                        # Track token usage
                        usage = data.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                        self._track_cost(input_tokens, output_tokens)

                        logger.debug(
                            f"GLM response generated",
                            extra={
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "model": self.model_name
                            }
                        )

                        return content

            except asyncio.TimeoutError:
                logger.warning(f"GLM API timeout (attempt {attempt + 1}/{retries})")
                last_error = "Timeout"
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(f"GLM API client error (attempt {attempt + 1}/{retries}): {e}")
                last_error = str(e)
                await asyncio.sleep(2 ** attempt)

        # All retries exhausted
        raise RuntimeError(f"GLM API failed after {retries} attempts: {last_error}")

    def _track_cost(self, input_tokens: int, output_tokens: int):
        """
        Track API cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # Convert from RMB to USD (approximate exchange rate)
        rmb_to_usd = 0.14

        input_cost = (input_tokens / 1_000_000) * self.input_price * rmb_to_usd
        output_cost = (output_tokens / 1_000_000) * self.output_price * rmb_to_usd
        total_cost = input_cost + output_cost

        self.cost_tracker.track_cost(
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost
        )

        logger.debug(
            f"GLM cost tracked",
            extra={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": total_cost
            }
        )

    async def health_check(self) -> bool:
        """
        Check if GLM API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = await self.generate("Hello", max_tokens=5)
            return bool(response)
        except Exception as e:
            logger.error(f"GLM health check failed: {e}")
            return False
