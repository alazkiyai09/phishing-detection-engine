"""
Mock LLM backend for testing without API calls.
Returns deterministic responses based on input hashing.
"""
import hashlib
import time
import json
from typing import Optional
from .base_llm import BaseLLM, LLMResponse


class MockLLM(BaseLLM):
    """
    Mock LLM backend that returns deterministic responses.
    Useful for testing without making actual API calls.
    """

    def __init__(self, model_name: str = "mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.response_delay_ms = kwargs.get("response_delay_ms", 0)
        self.responses_file = kwargs.get("responses_file")
        self.predefined_responses = self._load_predefined_responses()

    def _load_predefined_responses(self) -> dict:
        """Load predefined responses from a file if specified."""
        if self.responses_file:
            try:
                with open(self.responses_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return {}

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a mock response.
        Responses are deterministic based on the prompt hash.
        """
        start_time = time.time()

        # Simulate API delay if configured
        if self.response_delay_ms > 0:
            import asyncio
            await asyncio.sleep(self.response_delay_ms / 1000.0)

        # Check for predefined response first
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        if prompt_hash in self.predefined_responses:
            response_content = self.predefined_responses[prompt_hash]
        else:
            # Generate deterministic mock response based on prompt
            response_content = self._generate_mock_response(prompt, temperature)

        latency_ms = (time.time() - start_time) * 1000
        tokens_used = self.count_tokens(response_content)

        return LLMResponse(
            content=response_content,
            model=self.model_name,
            tokens_used=tokens_used,
            prompt_tokens=self.count_tokens(prompt),
            completion_tokens=tokens_used,
            latency_ms=latency_ms,
            metadata={"mock": True, "temperature": temperature}
        )

    def _generate_mock_response(self, prompt: str, temperature: float) -> str:
        """Generate a mock response based on prompt analysis."""
        prompt_lower = prompt.lower()

        # Determine if this is asking for phishing analysis
        if "phishing" in prompt_lower or "suspicious" in prompt_lower:
            # Check for indicators in the prompt
            if any(word in prompt_lower for word in ["urgent", "verify", "suspended", "click", "password", "account"]):
                return self._mock_phishing_response()
            else:
                return self._mock_legitimate_response()

        # URL analysis
        if "url" in prompt_lower and "analyze" in prompt_lower:
            return self._mock_url_analysis(prompt)

        # Header analysis
        if "header" in prompt_lower or "spf" in prompt_lower or "dkim" in prompt_lower:
            return self._mock_header_analysis(prompt)

        # Content analysis
        if "content" in prompt_lower or "text" in prompt_lower or "body" in prompt_lower:
            return self._mock_content_analysis(prompt)

        # Default response
        return self._mock_default_response()

    def _mock_phishing_response(self) -> str:
        """Mock response identifying phishing."""
        return json.dumps({
            "is_phishing": True,
            "confidence": 0.85,
            "reasoning": "This email contains multiple indicators of phishing including urgency tactics and requests for sensitive information.",
            "evidence": [
                "Urgent language: 'immediate action required'",
                "Request for credentials: 'verify your password'",
                "Suspicious URL: 'http://secure-login-verify.com'"
            ]
        }, indent=2)

    def _mock_legitimate_response(self) -> str:
        """Mock response identifying legitimate email."""
        return json.dumps({
            "is_phishing": False,
            "confidence": 0.9,
            "reasoning": "This email appears to be legitimate. It comes from a known domain and does not contain suspicious indicators.",
            "evidence": [
                "Sender domain: 'company.com' is legitimate",
                "No urgency indicators detected",
                "Professional tone and formatting"
            ]
        }, indent=2)

    def _mock_url_analysis(self, prompt: str) -> str:
        """Mock URL analysis response."""
        return json.dumps({
            "is_phishing": True,
            "confidence": 0.75,
            "reasoning": "The URLs in this email contain suspicious patterns including IP addresses and misleading domain names.",
            "evidence": [
                "URL uses IP address: http://192.168.1.1/login",
                "Typosquatting detected: 'g00gle.com' instead of 'google.com'",
                "Subdomain abuse: secure.verify.fake-bank.com"
            ]
        }, indent=2)

    def _mock_header_analysis(self, prompt: str) -> str:
        """Mock header analysis response."""
        return json.dumps({
            "is_phishing": True,
            "confidence": 0.8,
            "reasoning": "Email headers show authentication failures and suspicious routing.",
            "evidence": [
                "SPF verification failed",
                "No DKIM signature present",
                "Received headers show multiple unusual hops"
            ]
        }, indent=2)

    def _mock_content_analysis(self, prompt: str) -> str:
        """Mock content analysis response."""
        return json.dumps({
            "is_phishing": False,
            "confidence": 0.7,
            "reasoning": "Email content shows no strong social engineering patterns. Language is professional and non-urgent.",
            "evidence": [
                "No urgency language detected",
                "No credential requests",
                "Consistent branding and tone"
            ]
        }, indent=2)

    def _mock_default_response(self) -> str:
        """Default mock response."""
        return json.dumps({
            "is_phishing": False,
            "confidence": 0.5,
            "reasoning": "Insufficient information to make a determination.",
            "evidence": []
        }, indent=2)

    def count_tokens(self, text: str) -> int:
        """
        Mock token counting using word count approximation.
        Real tokenization is more complex, but this is sufficient for testing.
        """
        # Rough approximation: ~0.75 words per token
        return int(len(text.split()) / 0.75)
