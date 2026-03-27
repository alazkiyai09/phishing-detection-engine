"""
Main entry point for the Multi-Agent Phishing Detection System.
"""
import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multi_agent.models.email import EmailData, EmailHeaders, URL
from src.models.multi_agent.agents.coordinator import Coordinator
from src.models.multi_agent.llm.openai_backend import OpenAIBackend
from src.models.multi_agent.llm.local_backend import OllamaBackend
from src.models.multi_agent.llm.mock_backend import MockLLM
from src.models.multi_agent.cache.response_cache import ResponseCache
from src.models.multi_agent.utils.cost_tracker import CostTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def create_llm_backend(config: dict):
    """Create LLM backend based on configuration."""
    llm_config = config.get("llm", {})
    backend_type = llm_config.get("backend", "mock")

    if backend_type == "openai":
        logger.info(f"Using OpenAI backend with model: {llm_config.get('model', 'gpt-4')}")
        return OpenAIBackend(
            model_name=llm_config.get("model", "gpt-4"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
            timeout=llm_config.get("timeout", 30)
        )
    elif backend_type == "ollama":
        logger.info(f"Using Ollama backend with model: {llm_config.get('model', 'mistral:7b')}")
        return OllamaBackend(
            model_name=llm_config.get("model", "mistral:7b"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            timeout=llm_config.get("timeout", 120)
        )
    else:
        logger.info("Using Mock backend for testing")
        return MockLLM(
            model_name="mock-model",
            response_delay_ms=llm_config.get("mock", {}).get("response_delay_ms", 0)
        )


def create_sample_email_phishing() -> EmailData:
    """Create a sample phishing email for testing."""
    return EmailData(
        headers=EmailHeaders(
            subject="URGENT: Your Account Will Be Suspended",
            from_address="support@wellfarg0.com",  # Typosquatting
            to_addresses=["customer@example.com"],
            date=datetime.now(),
            message_id="<phishing123@fake.com>",
            spf="fail",
            dkim="none",
            dmarc="fail"
        ),
        body="""
        Dear Customer,

        Your Wells Fargo account has been compromised. Immediate action required!

        Click here to verify your account: http://wellfarg0.com/verify-login

        If you do not verify within 24 hours, your account will be permanently suspended.

        Please provide your password and social security number to verify your identity.

        ACT NOW TO PROTECT YOUR ACCOUNT!

        Urgent Notice from Wells Fargo Security Team
        """,
        urls=[
            URL(
                original="http://wellfarg0.com/verify-login",
                domain="wellfarg0.com",
                is_suspicious=True,
                suspicion_reasons=["Typosquatting of wellsfargo.com"]
            )
        ],
        email_id="sample-phishing-001",
        label=True
    )


def create_sample_email_legitimate() -> EmailData:
    """Create a sample legitimate email for testing."""
    return EmailData(
        headers=EmailHeaders(
            subject="Your Monthly Statement is Ready",
            from_address="notifications@wellsfargo.com",
            to_addresses=["john.doe@example.com"],
            date=datetime.now(),
            message_id="<stmt123@wellsfargo.com>",
            spf="pass",
            dkim="pass",
            dmarc="pass"
        ),
        body="""
        Dear John Doe,

        Your monthly statement for January 2025 is now available.

        Log in to your account at wellsfargo.com to view your statement.

        If you have any questions, please contact our customer service team.

        Best regards,
        Wells Fargo Bank
        """,
        urls=[
            URL(
                original="https://www.wellsfargo.com",
                domain="wellsfargo.com",
                is_suspicious=False,
                suspicion_reasons=[]
            )
        ],
        email_id="sample-legitimate-001",
        label=False
    )


async def analyze_single_email(
    email_path: Optional[str] = None,
    use_sample: str = "phishing",
    config_path: str = "config.yaml"
):
    """
    Analyze a single email.

    Args:
        email_path: Path to email file (JSON format)
        use_sample: Use sample email instead ("phishing" or "legitimate")
        config_path: Path to config file

    Returns:
        CoordinatorOutput
    """
    # Load configuration
    config = load_config(config_path)

    # Create components
    llm = create_llm_backend(config)

    cache = None
    if config.get("cache", {}).get("enabled", True):
        cache_config = config["cache"]
        cache = ResponseCache(
            max_size=cache_config.get("max_size", 1000),
            ttl_seconds=cache_config.get("ttl", 3600)
        )

    cost_tracker = None
    if config.get("cost_tracking", {}).get("enabled", True):
        cost_config = config["cost_tracking"]
        log_file = cost_config.get("log_file", "logs/cost_tracking.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        cost_tracker = CostTracker(log_file=log_file)

    # Create coordinator
    coord_config = config.get("coordinator", {})
    agent_config = config.get("agents", {})

    # Extract agent weights
    agent_weights = {}
    for agent_name, settings in agent_config.items():
        if settings.get("enabled", True):
            agent_weights[agent_name] = settings.get("weight", 1.0)

    coordinator = Coordinator(
        llm=llm,
        agent_weights=agent_weights,
        execution_mode=coord_config.get("execution_mode", "parallel"),
        voting_method=coord_config.get("voting_method", "weighted"),
        conflict_resolution_method=coord_config.get("conflict_resolution", "highest_confidence"),
        cache=cache,
        cost_tracker=cost_tracker,
        decision_threshold=coord_config.get("decision_threshold", 0.5)
    )

    # Load or create email
    if email_path:
        logger.info(f"Loading email from: {email_path}")
        with open(email_path, 'r') as f:
            email_data = json.load(f)
            email = EmailData(**email_data)
    else:
        logger.info(f"Using sample email: {use_sample}")
        if use_sample == "phishing":
            email = create_sample_email_phishing()
        else:
            email = create_sample_email_legitimate()

    # Analyze
    logger.info("Starting email analysis...")
    result = await coordinator.analyze_email(email)

    return result


def print_result(result) -> None:
    """Print analysis result in a formatted way."""
    print("\n" + "="*70)
    print("MULTI-AGENT PHISHING DETECTION - ANALYSIS RESULTS")
    print("="*70)

    # Final decision
    decision_str = "🚨 PHISHING" if result.is_phishing else "✅ LEGITIMATE"
    confidence_bar = "█" * int(result.confidence * 30)
    print(f"\nFinal Decision: {decision_str}")
    print(f"Confidence:    {result.confidence:.1%} |{confidence_bar:<30}|")
    print(f"Processing Time: {result.total_processing_time_ms:.1f}ms")
    print(f"Tokens Used: {result.llm_tokens_used:,}")
    print(f"Estimated Cost: ${result.estimated_cost_usd:.4f}")

    # Agent results
    print("\n" + "-"*70)
    print("AGENT RESULTS:")
    print("-"*70)

    for agent_name, output in result.agent_outputs.items():
        status = "✓" if not output.error else "✗"
        decision = "PHISHING" if output.is_phishing else "LEGITIMATE"
        print(f"\n{status} {agent_name}")
        print(f"   Decision: {decision} ({output.confidence:.0%})")
        print(f"   Latency: {output.processing_time_ms:.0f}ms")
        if output.error:
            print(f"   Error: {output.error}")

    # Voting result
    print("\n" + "-"*70)
    print("VOTING DETAILS:")
    print("-"*70)
    print(json.dumps(result.voting_result, indent=2))

    # Key evidence
    print("\n" + "-"*70)
    print("KEY EVIDENCE:")
    print("-"*70)
    for i, evidence in enumerate(result.key_evidence, 1):
        print(f"{i}. {evidence}")

    # Explanation
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    print(result.explanation)

    print("\n" + "="*70 + "\n")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Phishing Detection System"
    )
    parser.add_argument(
        "command",
        choices=["analyze", "demo"],
        help="Command to run"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Path to email file (JSON format)"
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["phishing", "legitimate"],
        default="phishing",
        help="Use sample email for demo"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        result = await analyze_single_email(
            email_path=args.email,
            use_sample=args.sample,
            config_path=args.config
        )

        print_result(result)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            logger.info(f"Results saved to: {args.output}")

    elif args.command == "demo":
        # Run demo with both phishing and legitimate samples
        print("\n🔍 RUNNING DEMO - Analyzing phishing email...")
        print("="*70)
        result_phishing = await analyze_single_email(
            use_sample="phishing",
            config_path=args.config
        )
        print_result(result_phishing)

        print("\n\n🔍 RUNNING DEMO - Analyzing legitimate email...")
        print("="*70)
        result_legitimate = await analyze_single_email(
            use_sample="legitimate",
            config_path=args.config
        )
        print_result(result_legitimate)


if __name__ == "__main__":
    asyncio.run(main())
