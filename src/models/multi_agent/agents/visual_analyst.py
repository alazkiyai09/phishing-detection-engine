"""Visual Analyst Agent - Placeholder for visual analysis of emails.

Note: This is currently a placeholder agent. Full implementation would require:
- HTML rendering engine (e.g., Selenium, Playwright)
- Visual similarity detection
- Screenshot comparison
"""

import logging
import re
import time
from typing import Dict, Any

from .base_agent import BaseAgent
from ..models.schemas import AgentOutput, EmailInput
from ..llm.base_llm import BaseLLM

try:
    from config.prompts import format_visual_prompt
except ImportError:
    def format_visual_prompt(subject, sender, body):
        return f"""Analyze the visual elements of this email for phishing indicators.

Subject: {subject}
Sender: {sender}

Body preview:
{body[:500]}

Provide your analysis in JSON format:
{{"is_phishing": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "evidence": ["list"]}}
"""


logger = logging.getLogger(__name__)


class VisualAnalyst(BaseAgent):
    """
    Analyzes visual elements of emails for phishing indicators.

    Current implementation (placeholder):
    - Basic HTML structure analysis
    - Checks for hidden elements
    - Detects form fields
    - Analyzes image references

    Future capabilities:
    - Screenshot rendering and comparison
    - Visual similarity to legitimate sites
    - Logo verification
    """

    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(llm=llm, agent_name="visual_analyst", **kwargs)

    async def analyze(self, email: EmailInput) -> AgentOutput:
        """Analyze email for visual phishing indicators."""
        start_time = time.time()

        try:
            # Run basic visual analysis (HTML-based)
            visual_result = self._analyze_visuals(email)

            # Build prompt
            prompt = self._build_prompt(email, visual_result)

            # Get LLM assessment
            response = await self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response.content)

            # Enhance with visual evidence
            parsed["evidence"].extend(visual_result["evidence"])

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
            logger.error(f"Visual Analyst failed: {e}")
            return await self._fallback_analysis(str(e))

    def _build_prompt(self, email: EmailInput, visual_result: Dict[str, Any] = None) -> str:
        """Build the visual analysis prompt."""
        return format_visual_prompt(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
        )

    def _analyze_visuals(self, email: EmailInput) -> Dict[str, Any]:
        """Perform basic visual analysis of email content."""
        result = {
            "has_forms": False,
            "has_password_fields": False,
            "hidden_elements": 0,
            "external_images": 0,
            "evidence": [],
        }

        body_lower = email.body.lower()

        # Check for forms
        if "<form" in body_lower:
            result["has_forms"] = True
            result["evidence"].append("Contains HTML forms")

            # Check form action
            form_actions = re.findall(
                r'<form[^>]*action=["\']([^"\']+)["\']', email.body, re.IGNORECASE
            )
            for action in form_actions:
                if "http" in action:
                    result["evidence"].append(f"Form submits to external URL")

        # Check for password fields
        if 'type="password"' in body_lower or "type='password'" in body_lower:
            result["has_password_fields"] = True
            result["evidence"].append("Contains password input field - HIGH RISK")

        # Check for hidden elements
        hidden_patterns = [
            'style="display:none"',
            "style='display:none'",
            'style="visibility:hidden"',
            "style='visibility:hidden'",
        ]

        for pattern in hidden_patterns:
            count = body_lower.count(pattern.lower())
            result["hidden_elements"] += count

        if result["hidden_elements"] > 5:
            result["evidence"].append(
                f"Excessive hidden elements ({result['hidden_elements']})"
            )

        # Check for external images
        img_sources = re.findall(r'<img[^>]*src=["\']([^"\']+)["\']', email.body, re.IGNORECASE)
        for src in img_sources:
            if src.startswith("http"):
                result["external_images"] += 1

                # Check for suspicious image sources
                if any(
                    suspicious in src.lower()
                    for suspicious in ["tracking", "pixel", "beacon", "trace"]
                ):
                    result["evidence"].append(f"Tracking pixel detected")

        # Check for iframes
        if "<iframe" in body_lower:
            result["evidence"].append("Contains iframe - potential phishing technique")

        # Check for redirects
        if 'meta http-equiv="refresh"' in body_lower or 'meta http-equiv=\'refresh\'' in body_lower:
            result["evidence"].append("Contains meta redirect - suspicious")

        # Check for JavaScript
        if "<script" in body_lower:
            script_count = body_lower.count("<script")
            if script_count > 3:
                result["evidence"].append(f"Excessive JavaScript ({script_count} scripts)")

        return result
