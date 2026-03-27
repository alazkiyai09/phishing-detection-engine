"""
Explanation generation for coordinator decisions.
"""
import logging
from typing import Dict, List
from ..models.agent_output import AgentOutput


logger = logging.getLogger(__name__)


def generate_explanation(
    outputs: Dict[str, AgentOutput],
    final_decision: bool,
    confidence: float,
    voting_details: Dict,
    conflict_resolution: str = None
) -> str:
    """
    Generate human-readable explanation for the final decision.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        final_decision: Final phishing classification
        confidence: Final confidence score
        voting_details: Details from voting process
        conflict_resolution: Description of conflict resolution if any

    Returns:
        Human-readable explanation
    """
    # Separate agents by their vote
    phishing_agents = []
    legitimate_agents = []
    failed_agents = []

    for agent_name, output in outputs.items():
        if output.error:
            failed_agents.append(agent_name)
        elif output.is_phishing:
            phishing_agents.append((agent_name, output.confidence))
        else:
            legitimate_agents.append((agent_name, output.confidence))

    # Build explanation
    explanation_parts = []

    # Opening statement
    if final_decision:
        explanation_parts.append(
            f"This email has been classified as **PHISHING** with {confidence:.0%} confidence."
        )
    else:
        explanation_parts.append(
            f"This email has been classified as **LEGITIMATE** with {confidence:.0%} confidence."
        )

    # Voting summary
    explanation_parts.append("\n### Agent Analysis:")

    if phishing_agents:
        explanation_parts.append(f"\n**Identified as Phishing:**")
        for agent_name, conf in sorted(phishing_agents, key=lambda x: -x[1]):
            explanation_parts.append(f"- {agent_name} (confidence: {conf:.0%})")

    if legitimate_agents:
        explanation_parts.append(f"\n**Identified as Legitimate:**")
        for agent_name, conf in sorted(legitimate_agents, key=lambda x: -x[1]):
            explanation_parts.append(f"- {agent_name} (confidence: {conf:.0%})")

    if failed_agents:
        explanation_parts.append(f"\n**Failed to Analyze:**")
        for agent_name in failed_agents:
            explanation_parts.append(f"- {agent_name}")

    # Add conflict resolution if applicable
    if conflict_resolution:
        explanation_parts.append(f"\n### Decision Process:")
        explanation_parts.append(f"\n{conflict_resolution}")

    # Key evidence
    explanation_parts.append("\n### Key Evidence:")

    # Collect and prioritize evidence
    all_evidence = []
    for agent_name, output in outputs.items():
        if output.error or not output.evidence:
            continue

        for evidence in output.evidence[:3]:  # Top 3 per agent
            all_evidence.append({
                "agent": agent_name,
                "text": evidence,
                "matches_decision": output.is_phishing == final_decision
            })

    # Prioritize evidence that matches final decision
    supporting_evidence = [e for e in all_evidence if e["matches_decision"]]
    opposing_evidence = [e for e in all_evidence if not e["matches_decision"]]

    if supporting_evidence:
        explanation_parts.append("\n**Supporting Evidence:**")
        for evidence in supporting_evidence[:5]:
            explanation_parts.append(f"- [{evidence['agent']}] {evidence['text']}")

    if opposing_evidence and len(opposing_evidence) <= 3:
        explanation_parts.append("\n**Contradictory Evidence:**")
        for evidence in opposing_evidence[:3]:
            explanation_parts.append(f"- [{evidence['agent']}] {evidence['text']}")

    # Confidence assessment
    explanation_parts.append("\n### Confidence Assessment:")

    if confidence >= 0.9:
        explanation_parts.append("Very high confidence - multiple strong indicators detected.")
    elif confidence >= 0.7:
        explanation_parts.append("High confidence - clear indicators detected.")
    elif confidence >= 0.5:
        explanation_parts.append("Moderate confidence - some indicators detected, manual review recommended.")
    else:
        explanation_parts.append("Low confidence - inconclusive evidence, manual review recommended.")

    return "\n".join(explanation_parts)


def extract_key_evidence(
    outputs: Dict[str, AgentOutput],
    final_decision: bool,
    top_n: int = 5
) -> List[str]:
    """
    Extract the most important evidence across all agents.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        final_decision: Final decision
        top_n: Number of top evidence items to return

    Returns:
        List of key evidence strings
    """
    evidence_scores = []

    for agent_name, output in outputs.items():
        if output.error or not output.evidence:
            continue

        # Score evidence based on:
        # - Matching final decision
        # - Agent confidence
        # - Evidence specificity (length)

        for evidence in output.evidence:
            score = 0.0

            # Boost if matches decision
            if output.is_phishing == final_decision:
                score += 1.0

            # Boost by agent confidence
            score += output.confidence

            # Boost by specificity (longer evidence is often more specific)
            score += min(len(evidence) / 100, 1.0)

            evidence_scores.append({
                "text": f"[{agent_name}] {evidence}",
                "score": score
            })

    # Sort by score and return top N
    evidence_scores.sort(key=lambda x: x["score"], reverse=True)
    return [e["text"] for e in evidence_scores[:top_n]]


def summarize_agent_performance(outputs: Dict[str, AgentOutput]) -> str:
    """
    Generate a summary of agent performance.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput

    Returns:
        Summary string
    """
    summary_parts = ["### Agent Performance Summary:\n"]

    for agent_name, output in outputs.items():
        if output.error:
            summary_parts.append(f"**{agent_name}**: FAILED - {output.error}")
        else:
            decision = "PHISHING" if output.is_phishing else "LEGITIMATE"
            summary_parts.append(
                f"**{agent_name}**: {decision} (confidence: {output.confidence:.0%}, "
                f"latency: {output.processing_time_ms:.0f}ms)"
            )

    return "\n".join(summary_parts)
