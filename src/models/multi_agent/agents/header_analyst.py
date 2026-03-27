"""Header Analyst Agent - Analyzes email headers for spoofing and authentication issues."""

import logging
import re
import time
from typing import Dict, Any

from .base_agent import BaseAgent
from ..models.schemas import AgentOutput, EmailInput
from ..llm.base_llm import BaseLLM

try:
    from config.prompts import format_header_prompt
except ImportError:
    def format_header_prompt(headers, subject, sender):
        headers_text = "\n".join(f"{k}: {v}" for k, v in headers.items())
        return f"""Analyze the following email headers for phishing indicators.

Subject: {subject}
Sender: {sender}

Headers:
{headers_text}

Provide your analysis in JSON format:
{{"is_phishing": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "evidence": ["list"]}}
"""


logger = logging.getLogger(__name__)


class HeaderAnalyst(BaseAgent):
    """
    Analyzes email headers for phishing indicators.

    Detection capabilities:
    - SPF (Sender Policy Framework) validation
    - DKIM (DomainKeys Identified Mail) validation
    - DMARC validation
    - Email spoofing detection
    - From/Reply-To mismatch detection
    """

    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(llm=llm, agent_name="header_analyst", **kwargs)

    async def analyze(self, email: EmailInput) -> AgentOutput:
        """Analyze email headers for authentication issues."""
        start_time = time.time()

        try:
            # Run header analysis
            header_result = self._analyze_headers(email)

            # Build prompt
            prompt = self._build_prompt(email, header_result)

            # Get LLM assessment
            response = await self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response.content)

            # Enhance with header evidence
            parsed["evidence"].extend(header_result["evidence"])

            # Adjust confidence based on authentication failures
            if header_result["auth_failures"] > 0:
                parsed["confidence"] = min(parsed["confidence"] + 0.2, 1.0)

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
            logger.error(f"Header Analyst failed: {e}")
            return await self._fallback_analysis(str(e))

    def _build_prompt(self, email: EmailInput, header_result: Dict[str, Any] = None) -> str:
        """Build the header analysis prompt."""
        return format_header_prompt(
            headers=email.headers,
            subject=email.subject,
            sender=email.sender,
        )

    def _analyze_headers(self, email: EmailInput) -> Dict[str, Any]:
        """Perform detailed header analysis."""
        result = {
            "auth_failures": 0,
            "from_reply_mismatch": False,
            "evidence": [],
        }

        headers = email.headers

        # Check SPF
        spf_value = headers.get("Received-SPF", headers.get("SPF", ""))
        if spf_value:
            if "fail" in spf_value.lower():
                result["auth_failures"] += 1
                result["evidence"].append(f"SPF validation failed")
            elif "pass" not in spf_value.lower():
                result["evidence"].append(f"SPF not passing: {spf_value[:100]}")
        else:
            result["evidence"].append("No SPF header present")

        # Check DKIM
        dkim_value = headers.get("DKIM-Signature", "")
        if not dkim_value:
            result["auth_failures"] += 1
            result["evidence"].append("No DKIM signature present")

        # Check From vs Reply-To mismatch
        from_address = email.sender
        reply_to = headers.get("Reply-To", "")

        if reply_to and from_address != reply_to:
            from_domain = from_address.split("@")[-1].lower() if "@" in from_address else ""
            reply_domain = reply_to.split("@")[-1].lower() if "@" in reply_to else ""

            if from_domain and reply_domain and from_domain != reply_domain:
                result["from_reply_mismatch"] = True
                result["evidence"].append(f"From/Reply-To mismatch: {from_address} vs {reply_to}")

        # Check Authentication-Results header
        auth_results = headers.get("Authentication-Results", "")
        if auth_results:
            if "fail" in auth_results.lower():
                result["auth_failures"] += 1
                result["evidence"].append(f"Authentication failures detected")

        return result
