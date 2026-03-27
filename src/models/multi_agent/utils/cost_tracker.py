"""
Token usage and cost tracking for LLM API calls.
"""
import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track LLM token usage and costs.

    Provides:
    - Real-time token counting
    - Cost estimation
    - Per-agent breakdown
    - Persistent logging
    """

    # Default pricing per 1K tokens (as of 2024)
    DEFAULT_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "mistral:7b": {"input": 0.0, "output": 0.0},  # Local model, free
        "mock-model": {"input": 0.0, "output": 0.0}
    }

    def __init__(
        self,
        log_file: Optional[str] = None,
        pricing: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize cost tracker.

        Args:
            log_file: Optional file to log costs to
            pricing: Custom pricing per model
        """
        self.log_file = log_file
        self.pricing = pricing or self.DEFAULT_PRICING

        # Statistics
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

        # Per-agent tracking
        self.agent_stats = {}

        # Request history
        self.history = []

    def track_request(
        self,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a single LLM request.

        Args:
            agent_name: Name of the agent making the request
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            latency_ms: Request latency
            metadata: Additional metadata

        Returns:
            Statistics about this request
        """
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        model_pricing = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (prompt_tokens / 1000.0) * model_pricing["input"]
        output_cost = (completion_tokens / 1000.0) * model_pricing["output"]
        total_cost = input_cost + output_cost

        # Update totals
        self.total_tokens += total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += total_cost

        # Update per-agent stats
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
                "latency_ms": []
            }

        self.agent_stats[agent_name]["requests"] += 1
        self.agent_stats[agent_name]["tokens"] += total_tokens
        self.agent_stats[agent_name]["cost"] += total_cost
        self.agent_stats[agent_name]["latency_ms"].append(latency_ms)

        # Record in history
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        self.history.append(entry)

        # Log to file if configured
        if self.log_file:
            self._log_to_file(entry)

        logger.info(
            f"LLM Request - Agent: {agent_name}, Model: {model}, "
            f"Tokens: {total_tokens}, Cost: ${total_cost:.4f}, "
            f"Latency: {latency_ms:.1f}ms"
        )

        return entry

    def get_total_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        avg_latency = []
        for agent_stats in self.agent_stats.values():
            avg_latency.extend(agent_stats["latency_ms"])

        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "total_requests": len(self.history),
            "average_latency_ms": sum(avg_latency) / len(avg_latency) if avg_latency else 0.0
        }

    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-agent statistics."""
        result = {}
        for agent_name, stats in self.agent_stats.items():
            avg_latency = sum(stats["latency_ms"]) / len(stats["latency_ms"]) if stats["latency_ms"] else 0.0
            result[agent_name] = {
                "requests": stats["requests"],
                "tokens": stats["tokens"],
                "cost_usd": round(stats["cost"], 4),
                "average_latency_ms": round(avg_latency, 2),
                "avg_tokens_per_request": stats["tokens"] / stats["requests"] if stats["requests"] > 0 else 0
            }
        return result

    def _log_to_file(self, entry: Dict[str, Any]) -> None:
        """Append entry to log file."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to cost log file: {e}")

    def export_history(self, filepath: str) -> None:
        """Export full history to JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump({
                    "summary": self.get_total_stats(),
                    "by_agent": self.get_agent_stats(),
                    "history": self.history
                }, f, indent=2)
            logger.info(f"Exported cost tracking history to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export history: {e}")

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.agent_stats = {}
        self.history = []
        logger.info("Reset cost tracking statistics")

    def print_summary(self) -> None:
        """Print a summary of tracked costs."""
        stats = self.get_total_stats()
        agent_stats = self.get_agent_stats()

        print("\n" + "="*60)
        print("COST TRACKING SUMMARY")
        print("="*60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"  - Prompt tokens: {stats['total_prompt_tokens']:,}")
        print(f"  - Completion tokens: {stats['total_completion_tokens']:,}")
        print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"Average Latency: {stats['average_latency_ms']:.1f}ms")
        print("\nPer-Agent Breakdown:")
        print("-"*60)

        for agent_name, agent_stat in agent_stats.items():
            print(f"{agent_name}:")
            print(f"  Requests: {agent_stat['requests']}")
            print(f"  Tokens: {agent_stat['tokens']:,}")
            print(f"  Cost: ${agent_stat['cost_usd']:.4f}")
            print(f"  Avg Latency: {agent_stat['average_latency_ms']:.1f}ms")

        print("="*60 + "\n")
