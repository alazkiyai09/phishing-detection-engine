"""Basic usage example for the multi-agent phishing detector."""

import asyncio
import logging

from src.models.schemas import EmailInput
from src.llm.mock_backend import MockLLM
from src.agents.coordinator import Coordinator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run a basic phishing detection example."""
    # Sample phishing email
    phishing_email = EmailInput(
        subject="URGENT: Your Account Will Be Suspended",
        sender="support@secure-verification.com",
        body="""
Dear Customer,

Your account has been flagged for unusual activity and will be suspended
within 24 hours unless you verify your identity immediately.

Click here to verify: http://secure-verification-confirm.com/login

Please enter your password and Social Security Number to confirm your account.

Failure to verify will result in permanent account closure.

Sincerely,
Security Team
""",
        urls=["http://secure-verification-confirm.com/login"],
        headers={
            "Received-SPF": "fail",
            "Reply-To": "phisher@scam.com",
        },
    )

    # Sample legitimate email
    legitimate_email = EmailInput(
        subject="Your order has been shipped",
        sender="shipping@amazon.com",
        body="""
Hello,

Your order #123-4567890 has been shipped and is on its way.

Estimated delivery: May 15 - May 17

Track your package: https://amazon.com/track/1234567890

Thank you for shopping with us!

Amazon Shipping Team
""",
        urls=["https://amazon.com/track/1234567890"],
        headers={
            "Received-SPF": "pass",
            "DKIM-Signature": "valid",
        },
    )

    # Initialize LLM backend (using MockLLM for demonstration)
    llm = MockLLM(model_name="mock-model")

    # Initialize coordinator
    coordinator = Coordinator(llm=llm, execution_mode="parallel")

    # Analyze phishing email
    print("\n" + "=" * 60)
    print("ANALYZING POTENTIAL PHISHING EMAIL")
    print("=" * 60)
    phishing_result = await coordinator.analyze_email(phishing_email)
    print(f"\nVerdict: {phishing_result.verdict.upper()}")
    print(f"Confidence: {phishing_result.confidence:.2%}")
    print(f"\nExplanation:\n{phishing_result.explanation}")
    print(f"\nFinancial Indicators:")
    print(f"  - Bank Impersonation: {phishing_result.financial_indicators.bank_impersonation}")
    print(f"  - Wire Urgency: {phishing_result.financial_indicators.wire_urgency}")
    print(f"  - Credential Harvesting: {phishing_result.financial_indicators.credential_harvesting}")
    print(f"  - Account Threats: {phishing_result.financial_indicators.account_threats}")
    print(f"\nTotal Latency: {phishing_result.total_latency_ms:.1f}ms")

    # Analyze legitimate email
    print("\n" + "=" * 60)
    print("ANALYZING LEGITIMATE EMAIL")
    print("=" * 60)
    legitimate_result = await coordinator.analyze_email(legitimate_email)
    print(f"\nVerdict: {legitimate_result.verdict.upper()}")
    print(f"Confidence: {legitimate_result.confidence:.2%}")
    print(f"\nExplanation:\n{legitimate_result.explanation}")
    print(f"\nTotal Latency: {legitimate_result.total_latency_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
