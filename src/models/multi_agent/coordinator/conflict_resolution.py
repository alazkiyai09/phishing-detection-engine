"""
Conflict resolution when agents disagree.
"""
import logging
from typing import Dict, List, Tuple
from ..models.agent_output import AgentOutput


logger = logging.getLogger(__name__)


def analyze_agreement(outputs: Dict[str, AgentOutput]) -> Dict:
    """
    Analyze level of agreement between agents.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput

    Returns:
        Dictionary with agreement metrics
    """
    valid_outputs = {k: v for k, v in outputs.items() if not v.error}

    if not valid_outputs:
        return {
            "total_agents": 0,
            "valid_agents": 0,
            "agreement_ratio": 0.0,
            "conflict_level": "no_valid_agents"
        }

    phishing_votes = sum(1 for o in valid_outputs.values() if o.is_phishing)
    legitimate_votes = len(valid_outputs) - phishing_votes

    # Calculate agreement ratio
    # 1.0 = unanimous, 0.0 = split
    max_votes = max(phishing_votes, legitimate_votes)
    total_votes = len(valid_outputs)
    agreement_ratio = max_votes / total_votes if total_votes > 0 else 0.0

    # Determine conflict level
    if agreement_ratio >= 0.8:
        conflict_level = "low"
    elif agreement_ratio >= 0.6:
        conflict_level = "medium"
    else:
        conflict_level = "high"

    return {
        "total_agents": len(outputs),
        "valid_agents": len(valid_outputs),
        "phishing_votes": phishing_votes,
        "legitimate_votes": legitimate_votes,
        "agreement_ratio": agreement_ratio,
        "conflict_level": conflict_level
    }


def resolve_by_highest_confidence(
    outputs: Dict[str, AgentOutput],
    weights: Dict[str, float]
) -> Tuple[bool, float, str]:
    """
    Resolve conflict by choosing the agent with highest weighted confidence.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        weights: Dictionary of agent_name -> weight

    Returns:
        Tuple of (is_phishing, confidence, resolution_method)
    """
    max_score = -1.0
    decision = False
    chosen_agent = None

    # SAFE: Filter out failed agents first
    valid_outputs = {k: v for k, v in outputs.items() if not v.error}

    # SAFE: Early return if no valid outputs
    if not valid_outputs:
        return False, 0.0, "No valid agent outputs for conflict resolution"

    for agent_name, output in valid_outputs.items():
        weight = weights.get(agent_name, 1.0)
        score = weight * output.confidence

        if score > max_score:
            max_score = score
            decision = output.is_phishing
            chosen_agent = agent_name

    resolution = f"Resolved by highest weighted confidence: {chosen_agent} (score: {max_score:.3f})"
    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, max_score))
    return decision, confidence, resolution


def resolve_by_trusted_agent(
    outputs: Dict[str, AgentOutput],
    trusted_agents: List[str]
) -> Tuple[bool, float, str]:
    """
    Resolve conflict by deferring to trusted agents.

    If multiple trusted agents disagree, use their weighted confidence.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        trusted_agents: List of agent names to trust

    Returns:
        Tuple of (is_phishing, confidence, resolution_method)
    """
    trusted_outputs = {
        k: v for k, v in outputs.items()
        if k in trusted_agents and not v.error
    }

    if not trusted_outputs:
        # No trusted agents available, fall back to highest confidence
        return resolve_by_highest_confidence(outputs, {})

    # If unanimous among trusted agents
    trusted_phishing = sum(1 for o in trusted_outputs.values() if o.is_phishing)
    if trusted_phishing == len(trusted_outputs):
        avg_confidence = sum(o.confidence for o in trusted_outputs.values()) / len(trusted_outputs)
        resolution = f"Unanimous trusted agents: {', '.join(trusted_outputs.keys())}"
        return True, max(0.0, min(1.0, avg_confidence)), resolution
    elif trusted_phishing == 0:
        avg_confidence = sum(o.confidence for o in trusted_outputs.values()) / len(trusted_outputs)
        resolution = f"Unanimous trusted agents: {', '.join(trusted_outputs.keys())}"
        return False, max(0.0, min(1.0, avg_confidence)), resolution

    # Trusted agents disagree, use weighted confidence
    max_score = -1.0
    decision = False
    chosen_agent = None

    for agent_name, output in trusted_outputs.items():
        score = output.confidence
        if score > max_score:
            max_score = score
            decision = output.is_phishing
            chosen_agent = agent_name

    resolution = f"Trusted agent disagreement, used highest confidence: {chosen_agent}"
    return decision, max(0.0, min(1.0, max_score)), resolution


def resolve_by_evidence_overlap(
    outputs: Dict[str, AgentOutput]
) -> Tuple[bool, float, str]:
    """
    Resolve conflict by analyzing evidence overlap.

    Agents with supporting evidence from other agents get boosted.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput

    Returns:
        Tuple of (is_phishing, confidence, resolution_method)
    """
    # Collect all evidence
    all_evidence = {}
    for agent_name, output in outputs.items():
        if output.error:
            continue
        all_evidence[agent_name] = output.evidence

    # Count evidence overlap
    overlap_scores = {}
    for agent_name, evidence in all_evidence.items():
        overlap_score = 0
        for other_agent, other_evidence in all_evidence.items():
            if agent_name == other_agent:
                continue

            # Check for similar evidence
            for ev1 in evidence:
                for ev2 in other_evidence:
                    # Simple string overlap check
                    if ev1.lower() in ev2.lower() or ev2.lower() in ev1.lower():
                        overlap_score += 1
                        break

        overlap_scores[agent_name] = overlap_score

    # Find agent with most supporting evidence
    if not overlap_scores:
        return resolve_by_highest_confidence(outputs, {})

    best_agent = max(overlap_scores, key=overlap_scores.get)
    best_output = outputs[best_agent]

    resolution = f"Resolved by evidence overlap: {best_agent} (overlap score: {overlap_scores[best_agent]})"
    return best_output.is_phishing, best_output.confidence, resolution


def resolve_conflict(
    outputs: Dict[str, AgentOutput],
    weights: Dict[str, float],
    method: str = "highest_confidence",
    **kwargs
) -> Tuple[bool, float, str]:
    """
    Resolve conflicts when agents disagree.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        weights: Dictionary of agent_name -> weight
        method: Resolution method
        **kwargs: Additional parameters for specific methods

    Returns:
        Tuple of (is_phishing, confidence, resolution_description)
    """
    # Check if there's actually a conflict
    agreement = analyze_agreement(outputs)

    if agreement["conflict_level"] == "low":
        # Low conflict, use voting result directly
        from .voting import weighted_vote
        is_phishing, confidence, _ = weighted_vote(outputs, weights)
        return is_phishing, confidence, "Low conflict, used weighted voting"

    # High conflict, use resolution method
    if method == "highest_confidence":
        return resolve_by_highest_confidence(outputs, weights)
    elif method == "trusted_agent":
        trusted_agents = kwargs.get("trusted_agents", ["content_analyst", "url_analyst"])
        return resolve_by_trusted_agent(outputs, trusted_agents)
    elif method == "evidence_overlap":
        return resolve_by_evidence_overlap(outputs)
    else:
        raise ValueError(f"Unknown conflict resolution method: {method}")
