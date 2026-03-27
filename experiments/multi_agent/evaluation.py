"""
Evaluation script for measuring multi-agent system performance.
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

from src.models.email import EmailData
from src.agents.coordinator import Coordinator
from src.llm.mock_backend import MockLLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate the multi-agent phishing detection system.

    Metrics:
    - Per-agent accuracy, precision, recall, F1
    - Ensemble vs single-agent comparison
    - Latency analysis
    - Cost analysis
    """

    def __init__(self, coordinator: Coordinator):
        """
        Initialize evaluator.

        Args:
            coordinator: Coordinator instance to evaluate
        """
        self.coordinator = coordinator

    async def evaluate(
        self,
        test_emails: List[EmailData]
    ) -> Dict:
        """
        Run full evaluation on test dataset.

        Args:
            test_emails: List of emails with ground truth labels

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {len(test_emails)} emails...")

        results = {
            "total_emails": len(test_emails),
            "ensemble_results": [],
            "per_agent_results": {},
            "latency_analysis": {},
            "cost_analysis": {}
        }

        # Initialize per-agent tracking
        for agent_name in self.coordinator.agents.keys():
            results["per_agent_results"][agent_name] = {
                "correct": 0,
                "incorrect": 0,
                "predictions": [],
                "latencies": []
            }

        # Run evaluation
        start_time = time.time()

        for i, email in enumerate(test_emails):
            logger.info(f"Analyzing email {i+1}/{len(test_emails)}: {email.email_id}")

            # Run coordinator
            result = await self.coordinator.analyze_email(email)

            # Track ensemble result
            correct = result.is_phishing == email.label
            results["ensemble_results"].append({
                "email_id": email.email_id,
                "predicted": result.is_phishing,
                "actual": email.label,
                "confidence": result.confidence,
                "correct": correct,
                "latency_ms": result.total_processing_time_ms
            })

            # Track per-agent results
            for agent_name, agent_output in result.agent_outputs.items():
                if agent_output.error:
                    continue

                agent_correct = agent_output.is_phishing == email.label
                results["per_agent_results"][agent_name]["correct"] += int(agent_correct)
                results["per_agent_results"][agent_name]["incorrect"] += int(not agent_correct)
                results["per_agent_results"][agent_name]["predictions"].append({
                    "predicted": agent_output.is_phishing,
                    "actual": email.label,
                    "confidence": agent_output.confidence
                })
                results["per_agent_results"][agent_name]["latencies"].append(
                    agent_output.processing_time_ms
                )

        total_time = time.time() - start_time
        results["total_evaluation_time"] = total_time

        # Calculate metrics
        results["ensemble_metrics"] = self._calculate_metrics(
            [r["correct"] for r in results["ensemble_results"]],
            [r["predicted"] for r in results["ensemble_results"]],
            [r["actual"] for r in results["ensemble_results"]]
        )

        for agent_name, agent_data in results["per_agent_results"].items():
            agent_data["metrics"] = self._calculate_metrics(
                [agent_data["correct"]] * agent_data["correct"],
                [p["predicted"] for p in agent_data["predictions"]],
                [p["actual"] for p in agent_data["predictions"]]
            )
            agent_data["avg_latency_ms"] = np.mean(agent_data["latencies"]) if agent_data["latencies"] else 0.0

        # Latency analysis
        results["latency_analysis"] = self._analyze_latency(results)

        return results

    def _calculate_metrics(
        self,
        correct_flags: List[bool],
        predictions: List[bool],
        actuals: List[bool]
    ) -> Dict:
        """Calculate classification metrics."""
        total = len(predictions)
        if total == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Basic metrics
        correct = sum(correct_flags)
        accuracy = correct / total

        # Precision, Recall, F1
        tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
        fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
        fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    def _analyze_latency(self, results: Dict) -> Dict:
        """Analyze latency across all agents."""
        latencies = [r["latency_ms"] for r in results["ensemble_results"]]

        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "std": np.std(latencies)
        }

    def print_report(self, results: Dict) -> None:
        """Print evaluation report."""
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)

        print(f"\nTotal Emails Analyzed: {results['total_emails']}")
        print(f"Total Evaluation Time: {results['total_evaluation_time']:.2f}s")

        # Ensemble results
        print("\n" + "-"*80)
        print("ENSEMBLE PERFORMANCE")
        print("-"*80)
        metrics = results["ensemble_metrics"]
        print(f"Accuracy:  {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall:    {metrics['recall']:.2%}")
        print(f"F1 Score:  {metrics['f1']:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")

        # Per-agent results
        print("\n" + "-"*80)
        print("PER-AGENT PERFORMANCE")
        print("-"*80)
        print(f"{'Agent':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg Latency':>12}")
        print("-"*80)

        # Sort by accuracy
        agent_results = results["per_agent_results"]
        sorted_agents = sorted(
            agent_results.items(),
            key=lambda x: x[1]["metrics"]["accuracy"],
            reverse=True
        )

        for agent_name, agent_data in sorted_agents:
            metrics = agent_data["metrics"]
            avg_latency = agent_data.get("avg_latency_ms", 0.0)
            print(f"{agent_name:<25} {metrics['accuracy']:>10.2%} {metrics['precision']:>10.2%} "
                  f"{metrics['recall']:>10.2%} {metrics['f1']:>10.2%} {avg_latency:>10.1f}ms")

        # Latency analysis
        print("\n" + "-"*80)
        print("LATENCY ANALYSIS")
        print("-"*80)
        latency = results["latency_analysis"]
        print(f"Mean:   {latency['mean']:.1f}ms")
        print(f"Median: {latency['median']:.1f}ms")
        print(f"Min:    {latency['min']:.1f}ms")
        print(f"Max:    {latency['max']:.1f}ms")
        print(f"Std:    {latency['std']:.1f}ms")

        print("\n" + "="*80 + "\n")

    def save_results(self, results: Dict, filepath: str) -> None:
        """Save evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {filepath}")


def create_test_dataset() -> List[EmailData]:
    """Create a small test dataset for evaluation."""
    from src.models.email import EmailData, EmailHeaders, URL

    test_emails = []

    # Phishing emails
    phishing_cases = [
        {
            "subject": "URGENT: Verify Your Account Now",
            "from": "support@secure-login.com",
            "body": "Click here immediately to verify your password or your account will be suspended.",
            "urls": ["http://192.168.1.1/verify"],
            "label": True
        },
        {
            "subject": "Wire Transfer Required",
            "from": "ceo@company-update.com",
            "body": "Urgent: Need immediate wire transfer. Please provide account details.",
            "urls": ["http://bank-verify.com/wire"],
            "label": True
        },
        {
            "subject": "Your Account is Compromised",
            "from": "security@wellfarg0.com",
            "body": "Your Wells Fargo account has been hacked. Verify your SSN immediately.",
            "urls": ["http://wellfarg0.com/verify"],
            "label": True
        }
    ]

    # Legitimate emails
    legitimate_cases = [
        {
            "subject": "Your Monthly Statement",
            "from": "notifications@wellsfargo.com",
            "body": "Your monthly statement is ready. Log in to view it.",
            "urls": ["https://www.wellsfargo.com"],
            "label": False
        },
        {
            "subject": "Order Confirmation",
            "from": "shipping@amazon.com",
            "body": "Your order has been shipped and will arrive in 2 days.",
            "urls": ["https://www.amazon.com/orders"],
            "label": False
        },
        {
            "subject": "Meeting Tomorrow",
            "from": "colleague@company.com",
            "body": "Let's meet tomorrow at 2pm to discuss the project.",
            "urls": [],
            "label": False
        }
    ]

    # Create EmailData objects
    for i, case in enumerate(phishing_cases + legitimate_cases, 1):
        email = EmailData(
            headers=EmailHeaders(
                subject=case["subject"],
                from_address=case["from"],
                to_addresses=["user@example.com"],
                date=datetime.now()
            ),
            body=case["body"],
            urls=[URL(original=u, domain=u.split('/')[2]) for u in case["urls"]],
            email_id=f"test-{i:03d}",
            label=case["label"]
        )
        test_emails.append(email)

    return test_emails


async def main():
    """Run evaluation."""
    # Create coordinator with mock LLM
    llm = MockLLM(model_name="mock-model")
    coordinator = Coordinator(llm=llm)

    # Create test dataset
    test_emails = create_test_dataset()

    # Run evaluation
    evaluator = Evaluator(coordinator)
    results = await evaluator.evaluate(test_emails)

    # Print report
    evaluator.print_report(results)

    # Save results
    output_path = "experiments/evaluation_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
