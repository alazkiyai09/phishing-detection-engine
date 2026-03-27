"""
Financial Specialist Agent - Specializes in bank impersonation and financial fraud detection.
"""
import re
import logging
from typing import List

from .base_agent import BaseAgent
from ..models.email import EmailData
from ..models.agent_output import AgentOutput


logger = logging.getLogger(__name__)


class FinancialSpecialist(BaseAgent):
    """
    Specialized agent for financial fraud detection.

    Detection capabilities:
    - Bank impersonation detection
    - Wire transfer urgency patterns
    - Invoice/payment fraud
    - Account suspension scams
    - Financial credential harvesting
    - Known legitimate bank domain verification
    """

    def __init__(self, llm, **kwargs):
        # Extract legitimate_domains before passing to super
        legitimate_domains = kwargs.pop("legitimate_domains", {
            "chase.com", "wellsfargo.com", "bankofamerica.com",
            "citi.com", "usbank.com", "capitalone.com",
            "schwab.com", "fidelity.com"
        })

        super().__init__(
            llm=llm,
            agent_name="financial_specialist",
            agent_version="1.0",
            **kwargs
        )

        # Load legitimate bank domains from config
        self.legitimate_domains = set(legitimate_domains)

        # Financial urgency patterns
        self.wire_urgency_patterns = [
            r'immediate\s+wire\s+transfer',
            r'urgent\s+payment',
            r'wire\s+transfer\s+urgently',
            r'same[- ]day\s+wire',
            r'rush\s+payment',
            r'priority\s+wire'
        ]

        # Bank account compromise patterns
        self.account_compromise_patterns = [
            r'unusual\s+activity',
            r'account\s+compromised',
            r'unauthorized\s+access',
            r'account\s+suspended',
            r'account\s+locked',
            r'security\s+breach',
            r'fraud\s+detected'
        ]

        # Financial action requests
        self.action_patterns = [
            r'confirm\s+your\s+(banking|account)\s+details',
            r'update\s+your\s+(payment|banking)\s+information',
            r'verify\s+your\s+(account|identity)',
            r'click\s+to\s+(verify|confirm|update)',
            r'provide\s+your\s+(account|routing)\s+number'
        ]

        # Compile patterns
        self.compiled_wire = [re.compile(p, re.IGNORECASE) for p in self.wire_urgency_patterns]
        self.compiled_compromise = [re.compile(p, re.IGNORECASE) for p in self.account_compromise_patterns]
        self.compiled_actions = [re.compile(p, re.IGNORECASE) for p in self.action_patterns]

    async def analyze(self, email: EmailData) -> AgentOutput:
        """
        Analyze email for financial fraud patterns.

        Args:
            email: Email data

        Returns:
            AgentOutput with phishing assessment
        """
        import time
        start_time = time.time()

        try:
            # Run financial-specific heuristic analysis
            financial_result = self._financial_heuristic_analysis(email)

            # Build prompt with financial context
            prompt = self._build_prompt(email, financial_result)

            # Get LLM assessment
            response = await self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response.content, email)

            # Enhance with financial evidence
            parsed["evidence"].extend(financial_result["evidence"])

            # Boost confidence for high-risk patterns
            if financial_result.get("high_risk_detected"):
                parsed["confidence"] = min(parsed["confidence"] + 0.3, 1.0)

            processing_time = (time.time() - start_time) * 1000

            return AgentOutput(
                agent_name=self.agent_name,
                agent_version=self.agent_version,
                is_phishing=parsed["is_phishing"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                evidence=parsed["evidence"],
                processing_time_ms=processing_time,
                llm_tokens_used=response.tokens_used,
                timestamp=email.received_timestamp,
                metadata=financial_result
            )

        except Exception as e:
            logger.error(f"Financial Specialist failed: {e}")
            return await self._fallback_analysis(email, str(e))

    def _build_prompt(self, email: EmailData, financial_result: dict = None) -> str:
        """Build the financial fraud analysis prompt."""
        # SAFE: Handle missing @ in from_address
        from_address = email.headers.from_address
        if '@' in from_address:
            sender_domain = from_address.split('@')[-1].lower()
        else:
            logger.warning(f"Invalid from_address (no @): {from_address}")
            sender_domain = from_address.lower()

        prompt = """Analyze this email for financial fraud and bank impersonation.

Focus on detecting:
1. Bank impersonation: Is the sender pretending to be a legitimate financial institution?
2. Wire transfer fraud: Urgent requests for wire transfers or payments
3. Account compromise scams: Claims account is compromised/suspended
4. Credential harvesting: Requests for banking credentials, account numbers, SSN
5. Invoice fraud: Fake invoices or payment requests

LEGITIMATE BANK DOMAINS (for reference):
{}

EMAIL DETAILS:
From: {} (domain: {})
Subject: {}

URLs found:
{}

EMAIL BODY:
{}

""".format(
    ", ".join(sorted(self.legitimate_domains)),
    from_address,
    sender_domain,
    email.headers.subject,
    "\n".join([url.original for url in email.urls[:5]]) if email.urls else "None",
    email.body[:2000]
)

        # Add financial heuristic analysis
        if financial_result:
            prompt += f"\nFINANCIAL ANALYSIS:\n"
            prompt += f"- Bank impersonation detected: {financial_result['bank_impersonation']}\n"
            prompt += f"- Wire transfer urgency: {financial_result['wire_urgency']}\n"
            prompt += f"- Account compromise claims: {financial_result['account_compromise']}\n"
            prompt += f"- Financial action requests: {financial_result['action_requests']}\n"
            prompt += f"- Domain impersonation: {financial_result['domain_impersonation']}\n"

            if financial_result.get("matched_banks"):
                prompt += f"- Matched legitimate banks: {', '.join(financial_result['matched_banks'])}\n"

        prompt += """
Provide your analysis in JSON format:
{
  "is_phishing": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of financial fraud indicators detected",
  "evidence": ["Specific financial fraud patterns or bank impersonation attempts"]
}
"""
        return prompt

    def _financial_heuristic_analysis(self, email: EmailData) -> dict:
        """Run financial-specific heuristic checks."""
        result = {
            "bank_impersonation": False,
            "wire_urgency": False,
            "account_compromise": False,
            "action_requests": False,
            "domain_impersonation": False,
            "matched_banks": [],
            "high_risk_detected": False,
            "evidence": []
        }

        # Combine subject and body
        text = f"{email.headers.subject} {email.body}".lower()

        # SAFE: Handle missing @ in from_address
        from_address = email.headers.from_address
        if '@' in from_address:
            sender_domain = from_address.split('@')[-1].lower()
        else:
            logger.warning(f"Invalid from_address in heuristic analysis: {from_address}")
            sender_domain = from_address.lower()

        # Check for bank name mentions
        for bank_domain in self.legitimate_domains:
            bank_name = bank_domain.split('.')[0]
            if bank_name in text or bank_domain in text:
                result["matched_banks"].append(bank_name)

        # Check if sender domain is actually a legitimate bank
        if sender_domain in self.legitimate_domains:
            result["evidence"].append(f"Sender domain is legitimate: {sender_domain}")
        else:
            # Check for domain impersonation
            for legit_domain in self.legitimate_domains:
                if self._is_domain_impersonation(sender_domain, legit_domain):
                    result["domain_impersonation"] = True
                    result["bank_impersonation"] = True
                    result["evidence"].append(f"Domain impersonation: {sender_domain} mimics {legit_domain}")
                    result["high_risk_detected"] = True
                    break

        # Check wire transfer urgency
        wire_count = 0
        for pattern in self.compiled_wire:
            matches = pattern.findall(email.body)
            if matches:
                wire_count += len(matches)
                for match in matches[:2]:
                    result["evidence"].append(f"Wire transfer urgency: '{match}'")

        if wire_count > 0:
            result["wire_urgency"] = True
            result["high_risk_detected"] = True

        # Check account compromise claims
        compromise_count = 0
        for pattern in self.compiled_compromise:
            matches = pattern.findall(text)
            if matches:
                compromise_count += len(matches)
                for match in matches[:2]:
                    result["evidence"].append(f"Account compromise claim: '{match}'")

        if compromise_count > 0:
            result["account_compromise"] = True

        # Check action requests
        action_count = 0
        for pattern in self.compiled_actions:
            matches = pattern.findall(text)
            if matches:
                action_count += len(matches)
                for match in matches[:2]:
                    result["evidence"].append(f"Financial action request: '{match}'")

        if action_count > 0:
            result["action_requests"] = True

        # Check for financial keywords in sender name
        sender_name = email.headers.from_address.lower()
        financial_keywords = ['bank', 'finance', 'financial', 'credit', 'payment', 'secure']
        if any(keyword in sender_name for keyword in financial_keywords):
            if sender_domain not in self.legitimate_domains:
                result["bank_impersonation"] = True
                result["evidence"].append(f"Suspicious: Financial keywords in sender but non-legitimate domain")
                result["high_risk_detected"] = True

        # Check for specific high-risk patterns
        high_risk_patterns = [
            r'wire\s+transfer.*\$\d+[,\d]*',
            r'invoice.*\$\d+[,\d]*',
            r'account.*suspended.*verify',
            r'urgent.*banking'
        ]

        for pattern in high_risk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["high_risk_detected"] = True
                break

        return result

    def _detect_bank_impersonation(self, text: str, urls: list) -> float:
        """
        Detect bank impersonation attempts.

        Returns:
            Float in [0, 1] indicating likelihood of impersonation
        """
        impersonation_score = 0.0
        text_lower = text.lower()

        # Check for bank names
        for bank_domain in self.legitimate_domains:
            bank_name = bank_domain.split('.')[0]
            if bank_name in text_lower:
                impersonation_score += 0.2

        # Check for banking terminology
        banking_terms = [
            'bank account', 'routing number', 'account number',
            'wire transfer', 'swift code', 'iban'
        ]

        for term in banking_terms:
            if term in text_lower:
                impersonation_score += 0.1

        return min(impersonation_score, 1.0)

    def _detect_wire_urgency(self, text: str) -> float:
        """
        Detect wire transfer urgency patterns.

        Returns:
            Float in [0, 1] indicating urgency level
        """
        urgency_score = 0.0
        text_lower = text.lower()

        # Wire transfer urgency keywords
        urgency_terms = [
            'wire transfer', 'urgent payment', 'immediate payment',
            'same day wire', 'rush payment', 'priority transfer'
        ]

        for term in urgency_terms:
            if term in text_lower:
                urgency_score += 0.2

        # Check for dollar amounts with urgency
        if re.search(r'\$\d+[,\d]*.*(urgent|immediate|today|now)', text_lower):
            urgency_score += 0.3

        return min(urgency_score, 1.0)

    def _is_domain_impersonation(self, suspicious: str, legitimate: str) -> bool:
        """
        Check if suspicious domain is impersonating legitimate domain.

        Examples:
        - wellsfarg0.com impersonates wellsfargo.com
        - chase-security.com impersonates chase.com
        """
        # Remove TLD for comparison
        suspicious_base = suspicious.split('.')[0]
        legitimate_base = legitimate.split('.')[0]

        # Check for character substitution (0 for o, 1 for l/i, etc.)
        substitutions = {
            '0': 'o', '1': 'l', '1': 'i', '3': 'e',
            '5': 's', '7': 't', '8': 'b', '9': 'g'
        }

        test_suspicious = suspicious_base.lower()
        for char, sub in substitutions.items():
            test_suspicious = test_suspicious.replace(char, sub)

        if test_suspicious == legitimate_base.lower():
            return True

        # Check for common prefix with slight modification
        if test_suspicious.startswith(legitimate_base.lower()):
            # wellsfargo-security vs wellsfargo
            return True

        # Check for added words
        if legitimate_base.lower() in test_suspicious:
            # chase-secure vs chase
            return True

        return False
