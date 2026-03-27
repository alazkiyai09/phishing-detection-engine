"""
Base LLM backend interface for the multi-agent system.
Defines the contract that all LLM backends must implement.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.
    All implementations (OpenAI, Ollama, Mock) must inherit from this.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM backend.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters

        Returns:
            LLMResponse with generated content and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    async def generate_structured(
        self,
        prompt: str,
        output_format: dict,
        temperature: float = 0.0,
        **kwargs
    ) -> dict:
        """
        Generate structured JSON output from the LLM.
        This is a convenience method that adds format instructions to the prompt.

        Args:
            prompt: The base prompt
            output_format: Dictionary describing expected output structure
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Parsed dictionary with LLM response
        """
        # Add format instructions to prompt
        format_instructions = self._get_format_instructions(output_format)
        full_prompt = f"{prompt}\n\n{format_instructions}"

        response = await self.generate(full_prompt, temperature=temperature, **kwargs)
        return self._parse_structured_response(response.content, output_format)

    def _get_format_instructions(self, output_format: dict) -> str:
        """Generate format instructions for structured output."""
        return f"""
Please respond with a JSON object following this structure:
{output_format}

Your response must be valid JSON only, with no additional text.
"""

    def _parse_structured_response(self, response: str, output_format: dict) -> dict:
        """Parse structured response from LLM."""
        import json

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")

    def validate_response(self, response: LLMResponse) -> bool:
        """
        Validate that an LLM response meets minimum quality criteria.

        Args:
            response: The LLM response to validate

        Returns:
            True if valid, False otherwise
        """
        if not response.content or len(response.content.strip()) == 0:
            return False
        return True

    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            "model_name": self.model_name,
            "backend": self.__class__.__name__,
            "config": self.config
        }
