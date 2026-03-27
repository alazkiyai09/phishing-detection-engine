"""Base agent class that all specialized agents inherit from.

Defines the standard interface and common functionality for all agents.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..models.schemas import AgentOutput, EmailInput
from ..llm.base_llm import BaseLLM, LLMResponse


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all analysis agents.

    All agents must inherit from this class and implement:
    - analyze(): Main analysis method
    - _build_prompt(): Construct the LLM prompt
    """

    def __init__(
        self,
        llm: BaseLLM,
        agent_name: str,
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        """Initialize the agent.

        Args:
            llm: LLM backend to use for analysis
            agent_name: Unique name for this agent
            temperature: Sampling temperature for LLM
            max_retries: Number of retries on LLM failure
        """
        self.llm = llm
        self.agent_name = agent_name
        self.temperature = temperature
        self.max_retries = max_retries

    @abstractmethod
    async def analyze(self, email: EmailInput) -> AgentOutput:
        """Analyze an email and produce an AgentOutput.

        Args:
            email: The email to analyze

        Returns:
            AgentOutput with classification, confidence, reasoning, and evidence
        """
        pass

    @abstractmethod
    def _build_prompt(self, email: EmailInput) -> str:
        """Build the LLM prompt for this agent.

        Args:
            email: The email to analyze

        Returns:
            Prompt string to send to LLM
        """
        pass

    async def _call_llm(self, prompt: str, timeout: float = 30.0) -> LLMResponse:
        """Call the LLM with retry logic and timeout.

        Args:
            prompt: Prompt to send to LLM
            timeout: Timeout in seconds for each LLM call

        Returns:
            LLMResponse

        Raises:
            RuntimeError: If all retries fail
            asyncio.TimeoutError: If timeout occurs
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.llm.generate(
                        prompt,
                        temperature=self.temperature,
                    ),
                    timeout=timeout
                )
                return response
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"LLM call timed out (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt < self.max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt < self.max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts") from last_error

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with is_phishing, confidence, reasoning, evidence
        """
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            parsed = json.loads(response)
            return {
                "is_phishing": bool(parsed.get("is_phishing", False)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")),
                "evidence": list(parsed.get("evidence", [])),
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return fallback response
            return {
                "is_phishing": False,
                "confidence": 0.0,
                "reasoning": f"Failed to parse LLM response: {str(e)}",
                "evidence": [],
            }

    def _determine_verdict(self, is_phishing: bool, confidence: float) -> str:
        """Determine the verdict string based on analysis.

        Args:
            is_phishing: Whether the email is phishing
            confidence: Confidence score

        Returns:
            "phishing", "legitimate", or "suspicious"
        """
        if confidence < 0.5:
            return "suspicious"
        return "phishing" if is_phishing else "legitimate"

    async def _fallback_analysis(self, error: str) -> AgentOutput:
        """Provide fallback analysis when LLM fails.

        Args:
            error: Error message from the failure

        Returns:
            AgentOutput with fallback analysis
        """
        return AgentOutput(
            agent_name=self.agent_name,
            verdict="suspicious",
            confidence=0.0,
            reasoning=f"LLM analysis failed. Error: {error}",
            evidence=[],
            latency_ms=0.0,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "name": self.agent_name,
            "llm_model": self.llm.model_name,
            "temperature": self.temperature,
        }
