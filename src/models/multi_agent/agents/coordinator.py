"""Coordinator Agent - Orchestrates parallel agent execution and aggregates results."""

import asyncio
import logging
import time
from typing import Dict, List, Any

from ..models.schemas import AgentOutput, CoordinatorDecision, EmailInput, FinancialIndicators
from ..llm.base_llm import BaseLLM

# Try to import config from root level
try:
    from config.agent_config import (
        AGENT_WEIGHTS,
        CONFIDENCE_THRESHOLD,
        CONFLICT_RESOLUTION_THRESHOLD,
        COORDINATION_CONFIG,
    )
    from config.financial_config import BANK_NAMES, URGENCY_TERMS, CREDENTIAL_TERMS, THREAT_TERMS
except ImportError:
    # Fallback defaults
    AGENT_WEIGHTS = {
        "url_analyst": 1.2,
        "content_analyst": 1.0,
        "header_analyst": 1.1,
        "visual_analyst": 0.8,
    }
    CONFIDENCE_THRESHOLD = 0.7
    CONFLICT_RESOLUTION_THRESHOLD = 0.5
    COORDINATION_CONFIG = {
        "min_agents_required": 2,
        "unanimous_threshold": 0.9,
        "majority_threshold": 0.6,
        "tie_breaker": "url_analyst",
    }
    # Financial indicator patterns
    BANK_NAMES = ["chase", "wells fargo", "bank of america", "citi", "capital one"]
    URGENCY_TERMS = ["urgent", "immediate", "wire transfer", "payment due"]
    CREDENTIAL_TERMS = ["password", "account number", "verify", "ssn"]
    THREAT_TERMS = ["suspended", "closed", "legal action", "consequences"]

# Aggregation constants
SUSPICIOUS_WEIGHT_SPLIT = 0.5  # Weight split for phishing/legitimate when verdict is suspicious
HIGH_CONFLICT_THRESHOLD = 0.75  # Minimum agreement ratio to avoid conflict resolution

from .url_analyst import URLAnalyst
from .content_analyst import ContentAnalyst
from .header_analyst import HeaderAnalyst
from .visual_analyst import VisualAnalyst


logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinator agent that orchestrates all analysis agents.

    Responsibilities:
    - Parallel agent execution
    - Result aggregation via weighted voting
    - Conflict resolution
    - Final decision generation
    - Graceful degradation on failures
    """

    def __init__(
        self,
        llm: BaseLLM,
        agent_weights: Dict[str, float] = None,
        execution_mode: str = "parallel",
        continue_on_failure: bool = True,
    ):
        """Initialize the coordinator.

        Args:
            llm: LLM backend to use
            agent_weights: Dictionary of agent_name -> weight for voting
            execution_mode: "parallel" or "sequential"
            continue_on_failure: Continue if some agents fail
        """
        self.llm = llm
        self.execution_mode = execution_mode
        self.continue_on_failure = continue_on_failure
        self.agent_weights = agent_weights or AGENT_WEIGHTS

        # Initialize agents
        self.agents = {
            "url_analyst": URLAnalyst(llm=llm),
            "content_analyst": ContentAnalyst(llm=llm),
            "header_analyst": HeaderAnalyst(llm=llm),
            "visual_analyst": VisualAnalyst(llm=llm),
        }

    async def analyze_email(self, email: EmailInput) -> CoordinatorDecision:
        """Analyze an email using all agents.

        Args:
            email: Email data to analyze

        Returns:
            CoordinatorDecision with final decision and explanation
        """
        start_time = time.time()

        # Run agents
        agent_outputs = await self._run_agents(email)

        # Aggregate results
        verdict, confidence, voting_details = self._aggregate_results(agent_outputs)

        # Resolve conflicts if necessary
        conflicts_resolved = []
        if self._has_high_conflict(agent_outputs):
            verdict, confidence, conflicts_resolved = self._resolve_conflicts(agent_outputs)

        # Generate explanation
        explanation = self._generate_explanation(agent_outputs, verdict, confidence, voting_details)

        # Extract financial indicators
        financial_indicators = self._extract_financial_indicators(agent_outputs, email)

        total_latency_ms = (time.time() - start_time) * 1000

        return CoordinatorDecision(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            agent_outputs=list(agent_outputs.values()),
            conflicts_resolved=conflicts_resolved,
            total_latency_ms=total_latency_ms,
            financial_indicators=financial_indicators,
        )

    async def _run_agents(self, email: EmailInput) -> Dict[str, AgentOutput]:
        """Run all agents on the email."""
        if self.execution_mode == "parallel":
            return await self._run_agents_parallel(email)
        else:
            return await self._run_agents_sequential(email)

    async def _run_agents_parallel(self, email: EmailInput) -> Dict[str, AgentOutput]:
        """Run agents in parallel."""
        logger.info(f"Running {len(self.agents)} agents in parallel")

        # Create tasks for all agents
        tasks = {agent_name: agent.analyze(email) for agent_name, agent in self.agents.items()}

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Process results
        outputs = {}
        for (agent_name, result) in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed: {result}")
                if self.continue_on_failure:
                    # Create error output with low confidence
                    outputs[agent_name] = AgentOutput(
                        agent_name=agent_name,
                        verdict="suspicious",
                        confidence=0.0,
                        reasoning=f"Agent failed: {str(result)}",
                        evidence=[],
                        latency_ms=0.0,
                    )
                else:
                    raise result
            else:
                outputs[agent_name] = result

        return outputs

    async def _run_agents_sequential(self, email: EmailInput) -> Dict[str, AgentOutput]:
        """Run agents sequentially."""
        logger.info(f"Running {len(self.agents)} agents sequentially")

        outputs = {}

        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Running {agent_name}")
                output = await agent.analyze(email)
                outputs[agent_name] = output
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                if self.continue_on_failure:
                    outputs[agent_name] = AgentOutput(
                        agent_name=agent_name,
                        verdict="suspicious",
                        confidence=0.0,
                        reasoning=f"Agent failed: {str(e)}",
                        evidence=[],
                        latency_ms=0.0,
                    )
                else:
                    raise

        return outputs

    def _aggregate_results(
        self, outputs: Dict[str, AgentOutput]
    ) -> tuple[str, float, Dict[str, Any]]:
        """Aggregate agent outputs using weighted voting.

        Returns:
            Tuple of (verdict, confidence, voting_details)
        """
        phishing_score = 0.0
        legitimate_score = 0.0
        total_weight = 0.0

        voting_details = {
            "agents_phishing": [],
            "agents_legitimate": [],
            "agents_suspicious": [],
        }

        for agent_name, output in outputs.items():
            weight = self.agent_weights.get(agent_name, 1.0)

            if output.verdict == "phishing":
                phishing_score += output.confidence * weight
                voting_details["agents_phishing"].append(agent_name)
            elif output.verdict == "legitimate":
                legitimate_score += (1 - output.confidence) * weight
                voting_details["agents_legitimate"].append(agent_name)
            else:  # suspicious
                # Split the weight using configurable constant
                phishing_score += output.confidence * weight * SUSPICIOUS_WEIGHT_SPLIT
                legitimate_score += (1 - output.confidence) * weight * SUSPICIOUS_WEIGHT_SPLIT
                voting_details["agents_suspicious"].append(agent_name)

            total_weight += weight

        # Normalize scores
        if total_weight > 0:
            phishing_score /= total_weight
            legitimate_score /= total_weight

        # Determine final verdict and confidence
        if phishing_score > legitimate_score:
            verdict = "phishing"
            confidence = phishing_score
        else:
            verdict = "legitimate"
            confidence = legitimate_score

        return verdict, confidence, voting_details

    def _has_high_conflict(self, outputs: Dict[str, AgentOutput]) -> bool:
        """Check if there's high conflict between agents."""
        verdicts = [output.verdict for output in outputs.values()]
        phishing_count = verdicts.count("phishing")
        legitimate_count = verdicts.count("legitimate")
        total = len(verdicts)

        if total < 2:
            return False

        # Check if agents disagree significantly
        ratio = max(phishing_count, legitimate_count) / total if total > 0 else 0
        return ratio < HIGH_CONFLICT_THRESHOLD  # Below threshold indicates conflict

    def _resolve_conflicts(self, outputs: Dict[str, AgentOutput]) -> tuple[str, float, List[str]]:
        """Resolve conflicts between disagreeing agents.

        Returns:
            Tuple of (verdict, confidence, conflict_resolutions)
        """
        conflict_resolutions = []

        # Find highest confidence agent
        highest_confidence_output = max(
            outputs.values(),
            key=lambda x: x.confidence if x.verdict != "suspicious" else 0,
        )

        # Trust the highest confidence agent
        conflict_resolutions.append(
            f"Trusting {highest_confidence_output.agent_name} "
            f"with highest confidence ({highest_confidence_output.confidence:.2f})"
        )

        return (
            highest_confidence_output.verdict,
            highest_confidence_output.confidence,
            conflict_resolutions,
        )

    def _generate_explanation(
        self,
        outputs: Dict[str, AgentOutput],
        verdict: str,
        confidence: float,
        voting_details: Dict[str, Any],
    ) -> str:
        """Generate explanation for the final decision."""
        lines = [
            f"Final Verdict: {verdict.upper()}",
            f"Overall Confidence: {confidence:.2%}",
            "",
            "Agent Breakdown:",
        ]

        for agent_name, output in outputs.items():
            lines.append(f"  - {agent_name}: {output.verdict} ({output.confidence:.2%})")

        lines.append("")
        lines.extend(["Key Evidence:", ""])

        # Collect all evidence
        all_evidence = []
        for output in outputs.values():
            all_evidence.extend(output.evidence)

        # Show unique evidence items
        seen = set()
        for evidence in all_evidence[:10]:  # Limit to 10 items
            if evidence not in seen:
                lines.append(f"  - {evidence}")
                seen.add(evidence)

        return "\n".join(lines)

    def _extract_financial_indicators(
        self, outputs: Dict[str, AgentOutput], email: EmailInput
    ) -> FinancialIndicators:
        """Extract financial domain-specific indicators."""
        indicators = FinancialIndicators()

        # Check content analyst for financial patterns
        content_output = outputs.get("content_analyst")
        if content_output:
            # Check for bank impersonation (using configurable bank list)
            text = f"{email.subject} {email.body}".lower()
            if any(bank in text for bank in BANK_NAMES):
                indicators.bank_impersonation = True

            # Check for urgency (using configurable term lists)
            if any(term in text for term in URGENCY_TERMS):
                indicators.wire_urgency = True

            # Check for credential harvesting
            if any(term in text for term in CREDENTIAL_TERMS):
                indicators.credential_harvesting = True

            # Check for account threats
            if any(term in text for term in THREAT_TERMS):
                indicators.account_threats = True

        return indicators
