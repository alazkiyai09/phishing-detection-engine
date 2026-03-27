"""Content Analyst Agent - Analyzes email text for social engineering tactics."""

import logging
import re
import time
from typing import List, Dict, Any

from .base_agent import BaseAgent
from ..models.schemas import AgentOutput, EmailInput
from ..llm.base_llm import BaseLLM

try:
    from config.prompts import format_content_prompt
except ImportError:
    def format_content_prompt(subject, sender, body):
        return f"""Analyze the following email content for phishing indicators.

Subject: {subject}
From: {sender}

Body:
{body}

Provide your analysis in JSON format:
{{"is_phishing": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "evidence": ["list"]}}
"""


logger = logging.getLogger(__name__)


class ContentAnalyst(BaseAgent):
    """
    Analyzes email content for phishing indicators.

    Detection capabilities:
    - Social engineering tactics (urgency, authority, intimidation)
    - Credential harvesting attempts
    - Requests for sensitive information
    - Generic greetings
    - Poor grammar/spelling
    """

    URGENCY_PATTERNS = [
        r"immediate\s+action\s+required",
        r"urgent\s+notice",
        r"act\s+now",
        r"expires?\s+soon",
        r"account\s+suspended",
        r"account\s+will\s+be\s+closed",
        r"immediate\s+attention",
    ]

    CREDENTIAL_PATTERNS = [
        r"password",
        r"PIN",
        r"social\s+security",
        r"credit\s+card",
        r"verify\s+your\s+(identity|information)",
        r"confirm\s+your\s+(account|details)",
        r"update\s+your\s+information",
    ]

    GENERIC_GREETINGS = [
        r"dear\s+customer",
        r"dear\s+user",
        r"dear\s+client",
        r"valued\s+customer",
    ]

    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(llm=llm, agent_name="content_analyst", **kwargs)

        # Compile patterns for efficiency
        self.compiled_urgency = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self.compiled_credentials = [re.compile(p, re.IGNORECASE) for p in self.CREDENTIAL_PATTERNS]
        self.compiled_greetings = [re.compile(p, re.IGNORECASE) for p in self.GENERIC_GREETINGS]

    async def analyze(self, email: EmailInput) -> AgentOutput:
        """Analyze email content for phishing patterns."""
        start_time = time.time()

        try:
            # Run heuristic analysis
            heuristic_result = self._heuristic_analysis(email)

            # Build prompt
            prompt = self._build_prompt(email, heuristic_result)

            # Get LLM assessment
            response = await self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response.content)

            # Enhance with heuristic evidence
            parsed["evidence"].extend(heuristic_result["evidence"])

            latency_ms = (time.time() - start_time) * 1000
            verdict = self._determine_verdict(parsed["is_phishing"], parsed["confidence"])

            return AgentOutput(
                agent_name=self.agent_name,
                verdict=verdict,
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                evidence=parsed["evidence"],
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Content Analyst failed: {e}")
            return await self._fallback_analysis(str(e))

    def _build_prompt(self, email: EmailInput, heuristic_result: Dict[str, Any] = None) -> str:
        """Build the content analysis prompt."""
        return format_content_prompt(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
        )

    def _heuristic_analysis(self, email: EmailInput) -> Dict[str, Any]:
        """Run heuristic checks on email content."""
        # Combine subject and body
        text = f"{email.subject} {email.body}"

        result = {
            "urgency_count": 0,
            "credential_count": 0,
            "has_generic_greeting": False,
            "evidence": [],
        }

        # Check urgency patterns
        for pattern in self.compiled_urgency:
            matches = pattern.findall(text)
            if matches:
                result["urgency_count"] += len(matches)
                for match in matches[:2]:
                    result["evidence"].append(f"Urgency pattern: '{match}'")

        # Check credential patterns
        for pattern in self.compiled_credentials:
            matches = pattern.findall(text)
            if matches:
                result["credential_count"] += len(matches)
                for match in matches[:2]:
                    result["evidence"].append(f"Credential request: '{match}'")

        # Check for generic greetings
        greeting_section = text[:200].lower()
        for pattern in self.compiled_greetings:
            if pattern.search(greeting_section):
                result["has_generic_greeting"] = True
                result["evidence"].append("Generic greeting detected")
                break

        # Check for all caps urgency
        if re.search(r"(?:URGENT|IMMEDIATE|ATTENTION|WARNING)", text):
            result["urgency_count"] += 1
            result["evidence"].append("All-caps urgency words")

        # Check for excessive punctuation
        if text.count("!!!") > 0 or text.count("???") > 0:
            result["urgency_count"] += 1
            result["evidence"].append("Excessive punctuation (!!!, ???)")

        return result
