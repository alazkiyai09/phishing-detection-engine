"""Evaluation script for comparing ensemble vs single-agent performance."""

import asyncio
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from src.models.schemas import EmailInput
from src.llm.mock_backend import MockLLM
from src.agents.coordinator import Coordinator
from src.agents.url_analyst import URLAnalyst
from src.agents.content_analyst import ContentAnalyst
from src.agents.header_analyst import HeaderAnalyst
from src.agents.visual_analyst import VisualAnalyst


# Sample emails for evaluation
TEST_EMAILS = [
    {
        "email": EmailInput(
            subject="Verify Your Account Immediately",
            sender="support@secure-login.net",
            body="Click here to verify your password.",
            urls=["http://secure-login.net/verify"],
            headers={"Received-SPF": "fail"},
        ),
        "expected": "phishing",
    },
    {
        "email": EmailInput(
            subject="Your weekly newsletter",
            sender="news@company.com",
            body="Here is your weekly update.",
            urls=["https://company.com/newsletter"],
            headers={"Received-SPF": "pass"},
        ),
        "expected": "legitimate",
    },
    {
        "email": EmailInput(
            subject="URGENT: Wire Transfer Request",
            sender="ceo@company-corp.com",
            body="Please wire $5000 to account 123456 immediately.",
            urls=[],
            headers={"Received-SPF": "neutral"},
        ),
        "expected": "phishing",
    },
    {
        "email": EmailInput(
            subject="Meeting tomorrow",
            sender="colleague@company.com",
            body="Let's meet at 2pm to discuss the project.",
            urls=[],
            headers={"Received-SPF": "pass"},
        ),
        "expected": "legitimate",
    },
]


async def evaluate_ensemble(llm):
    """Evaluate the full ensemble coordinator."""
    coordinator = Coordinator(llm=llm, execution_mode="parallel")

    results = []
    for test_case in TEST_EMAILS:
        start_time = time.time()
        decision = await coordinator.analyze_email(test_case["email"])
        latency = time.time() - start_time

        results.append(
            {
                "expected": test_case["expected"],
                "predicted": decision.verdict,
                "confidence": decision.confidence,
                "latency_ms": latency * 1000,
                "correct": decision.verdict == test_case["expected"],
            }
        )

    return results


async def evaluate_single_agent(llm, agent_class, agent_name):
    """Evaluate a single agent in isolation."""
    agent = agent_class(llm=llm)

    results = []
    for test_case in TEST_EMAILS:
        start_time = time.time()
        output = await agent.analyze(test_case["email"])
        latency = time.time() - start_time

        # Map agent verdict to binary
        predicted = "phishing" if output.verdict == "phishing" else "legitimate"

        results.append(
            {
                "agent": agent_name,
                "expected": test_case["expected"],
                "predicted": predicted,
                "confidence": output.confidence,
                "latency_ms": latency * 1000,
                "correct": predicted == test_case["expected"],
            }
        )

    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Calculate true positives, false positives, true negatives, false negatives
    tp = sum(1 for r in results if r["predicted"] == "phishing" and r["expected"] == "phishing")
    fp = sum(1 for r in results if r["predicted"] == "phishing" and r["expected"] == "legitimate")
    tn = sum(1 for r in results if r["predicted"] == "legitimate" and r["expected"] == "legitimate")
    fn = sum(1 for r in results if r["predicted"] == "legitimate" and r["expected"] == "phishing")

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_latency_ms": avg_latency,
        "avg_confidence": avg_confidence,
        "total_correct": correct,
        "total_tests": total,
    }


async def main():
    """Run evaluation comparing ensemble vs single agents."""
    llm = MockLLM(model_name="mock-model")

    print("=" * 70)
    print("MULTI-AGENT PHISHING DETECTOR - EVALUATION")
    print("=" * 70)

    # Evaluate ensemble
    print("\n[1/5] Evaluating Ensemble Coordinator...")
    ensemble_results = await evaluate_ensemble(llm)
    ensemble_metrics = calculate_metrics(ensemble_results)

    # Evaluate individual agents
    agent_classes = [
        (URLAnalyst, "url_analyst"),
        (ContentAnalyst, "content_analyst"),
        (HeaderAnalyst, "header_analyst"),
        (VisualAnalyst, "visual_analyst"),
    ]

    agent_metrics = {}
    for i, (agent_class, agent_name) in enumerate(agent_classes, 2):
        print(f"[{i}/5] Evaluating {agent_name}...")
        results = await evaluate_single_agent(llm, agent_class, agent_name)
        agent_metrics[agent_name] = calculate_metrics(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nEnsemble Coordinator:")
    print(f"  Accuracy:    {ensemble_metrics['accuracy']:.2%}")
    print(f"  Precision:   {ensemble_metrics['precision']:.2%}")
    print(f"  Recall:      {ensemble_metrics['recall']:.2%}")
    print(f"  F1 Score:    {ensemble_metrics['f1_score']:.2%}")
    print(f"  Avg Latency: {ensemble_metrics['avg_latency_ms']:.1f}ms")
    print(f"  Avg Confidence: {ensemble_metrics['avg_confidence']:.2%}")

    print("\nIndividual Agents:")
    for agent_name, metrics in agent_metrics.items():
        print(f"\n{agent_name}:")
        print(f"  Accuracy:    {metrics['accuracy']:.2%}")
        print(f"  F1 Score:    {metrics['f1_score']:.2%}")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")

    # Compare ensemble vs best single agent
    best_agent = max(agent_metrics.items(), key=lambda x: x[1]["f1_score"])
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nEnsemble vs Best Single Agent ({best_agent[0]}):")
    print(f"  F1 Score Improvement: {ensemble_metrics['f1_score'] - best_agent[1]['f1_score']:.2%}")
    print(f"  Latency Overhead: {ensemble_metrics['avg_latency_ms'] - best_agent[1]['avg_latency_ms']:.1f}ms")

    # Save results to JSON
    output = {
        "ensemble": ensemble_metrics,
        "agents": agent_metrics,
    }

    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
