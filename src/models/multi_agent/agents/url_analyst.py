"""URL Analyst Agent - Analyzes URLs for suspicious patterns and phishing indicators."""

import logging
import time
from typing import List
from urllib.parse import urlparse

from .base_agent import BaseAgent
from ..models.schemas import AgentOutput, EmailInput
from ..llm.base_llm import BaseLLM

try:
    from config.prompts import format_url_prompt
except ImportError:
    # Fallback prompt formatting
    def format_url_prompt(urls, subject, sender, body_preview):
        urls_text = "\n".join(f"- {url}" for url in urls) if urls else "(No URLs)"
        return f"""Analyze the following URLs from an email for phishing indicators.

Subject: {subject}
Sender: {sender}

URLs:
{urls_text}

Body preview:
{body_preview}

Provide your analysis in JSON format:
{{"is_phishing": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "evidence": ["list"]}}
"""


logger = logging.getLogger(__name__)


class URLAnalyst(BaseAgent):
    """
    Analyzes URLs in emails for phishing indicators.

    Detection capabilities:
    - IP addresses instead of domain names
    - Typosquatting (e.g., g00gle.com)
    - Suspicious TLDs
    - Subdomain abuse
    - Brand impersonation in URLs
    """

    SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".club"}

    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(llm=llm, agent_name="url_analyst", **kwargs)
        self.legitimate_domains = kwargs.get(
            "legitimate_domains",
            {"google.com", "facebook.com", "amazon.com", "apple.com", "microsoft.com"},
        )

    async def analyze(self, email: EmailInput) -> AgentOutput:
        """Analyze URLs in the email."""
        start_time = time.time()

        try:
            # Extract URLs if not already present
            urls = email.urls if email.urls else self._extract_urls(email.body)

            if not urls:
                return AgentOutput(
                    agent_name=self.agent_name,
                    verdict="suspicious",
                    confidence=0.5,
                    reasoning="No URLs found in email. Cannot determine if phishing based on URLs.",
                    evidence=[],
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # Run heuristic analysis
            heuristic_result = self._heuristic_analysis(urls)

            # Build prompt for LLM
            prompt = self._build_prompt(email, heuristic_result)

            # Get LLM assessment
            response = await self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response.content)

            # Enhance evidence with heuristic findings
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
            logger.error(f"URL Analyst failed: {e}")
            return await self._fallback_analysis(str(e))

    def _build_prompt(self, email: EmailInput, heuristic_result: dict = None) -> str:
        """Build the URL analysis prompt."""
        return format_url_prompt(
            urls=email.urls if email.urls else self._extract_urls(email.body),
            subject=email.subject,
            sender=email.sender,
            body_preview=email.body[:500],
        )

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        import re

        url_pattern = r"https?://[^\s<>\"]+"
        return re.findall(url_pattern, text)

    def _heuristic_analysis(self, urls: List[str]) -> dict:
        """Run heuristic checks on URLs."""
        result = {
            "suspicious_count": 0,
            "evidence": [],
        }

        for url in urls:
            parsed = urlparse(url)

            # Check for IP address
            if self._is_ip_address(parsed.netloc):
                result["suspicious_count"] += 1
                result["evidence"].append(f"IP address in URL: {url}")

            # Check for suspicious TLD
            if any(url.lower().endswith(tld) for tld in self.SUSPICIOUS_TLDS):
                result["suspicious_count"] += 1
                result["evidence"].append(f"Suspicious TLD: {url}")

            # Check for typosquatting
            for legit_domain in self.legitimate_domains:
                if self._is_typosquat(parsed.netloc, legit_domain):
                    result["suspicious_count"] += 1
                    result["evidence"].append(f"Possible typosquatting: {parsed.netloc} vs {legit_domain}")

        return result

    def _is_ip_address(self, netloc: str) -> bool:
        """Check if netloc is an IP address."""
        import re

        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        return bool(re.match(ip_pattern, netloc.split(":")[0]))

    def _is_typosquat(self, domain: str, legitimate: str) -> bool:
        """Check if domain is a typosquat of legitimate domain."""
        if domain.lower() == legitimate.lower():
            return False

        # Simple edit distance check
        distance = self._levenshtein_distance(domain.lower(), legitimate.lower())
        return distance <= 2

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
