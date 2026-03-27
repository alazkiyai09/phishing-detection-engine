"""LLM backend implementations for the multi-agent system."""

from .base_llm import BaseLLM, LLMResponse
from .mock_backend import MockLLM
from .ollama_backend import OllamaBackend

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "MockLLM",
    "OllamaBackend",
]

# Try to import OpenAI backend (optional dependency)
try:
    from .openai_backend import OpenAIBackend
    __all__.append("OpenAIBackend")
except ImportError:
    pass
