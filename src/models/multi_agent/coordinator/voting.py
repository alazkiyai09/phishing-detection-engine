"""
Weighted voting logic for aggregating agent outputs.
"""
import logging
from typing import Dict, List, Tuple
from ..models.agent_output import AgentOutput


logger = logging.getLogger(__name__)


def weighted_vote(
    outputs: Dict[str, AgentOutput],
    weights: Dict[str, float]
) -> Tuple[bool, float, Dict]:
    """
    Perform weighted voting on agent outputs.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        weights: Dictionary of agent_name -> weight

    Returns:
        Tuple of (is_phishing, confidence, voting_details)
    """
    total_weight = 0.0
    phishing_weight = 0.0
    legitimate_weight = 0.0
    weighted_confidence_sum = 0.0

    votes = {}

    for agent_name, output in outputs.items():
        if output.error:
            # Skip failed agents
            logger.warning(f"Skipping failed agent {agent_name} in voting")
            continue

        weight = weights.get(agent_name, 1.0)
        total_weight += weight

        if output.is_phishing:
            phishing_weight += weight * output.confidence
            votes[agent_name] = {"vote": "phishing", "weight": weight, "confidence": output.confidence}
        else:
            legitimate_weight += weight * (1.0 - output.confidence)
            votes[agent_name] = {"vote": "legitimate", "weight": weight, "confidence": output.confidence}

        weighted_confidence_sum += weight * output.confidence

    # SAFE CHECK: Verify total_weight > 0 before division
    if total_weight == 0:
        return False, 0.0, {"error": "No valid agent outputs", "votes": votes}

    is_phishing = phishing_weight > legitimate_weight

    # SAFE: total_weight is now guaranteed > 0
    if is_phishing:
        confidence = phishing_weight / total_weight
    else:
        confidence = legitimate_weight / total_weight

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    voting_details = {
        "method": "weighted",
        "total_weight": total_weight,
        "phishing_weight": phishing_weight,
        "legitimate_weight": legitimate_weight,
        "votes": votes,
        "winner": "phishing" if is_phishing else "legitimate"
    }

    return is_phishing, confidence, voting_details


def majority_vote(
    outputs: Dict[str, AgentOutput]
) -> Tuple[bool, float, Dict]:
    """
    Perform simple majority voting.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput

    Returns:
        Tuple of (is_phishing, confidence, voting_details)
    """
    phishing_count = 0
    legitimate_count = 0
    total_count = 0

    votes = {}

    for agent_name, output in outputs.items():
        if output.error:
            continue

        total_count += 1
        if output.is_phishing:
            phishing_count += 1
            votes[agent_name] = "phishing"
        else:
            legitimate_count += 1
            votes[agent_name] = "legitimate"

    if total_count == 0:
        return False, 0.0, {"error": "No valid agent outputs"}

    is_phishing = phishing_count > legitimate_count
    confidence = max(phishing_count, legitimate_count) / total_count

    voting_details = {
        "method": "majority",
        "total_agents": total_count,
        "phishing_votes": phishing_count,
        "legitimate_votes": legitimate_count,
        "votes": votes,
        "winner": "phishing" if is_phishing else "legitimate"
    }

    return is_phishing, confidence, voting_details


def confidence_weighted_vote(
    outputs: Dict[str, AgentOutput],
    weights: Dict[str, float]
) -> Tuple[bool, float, Dict]:
    """
    Vote where each agent's contribution is weighted by their confidence.

    This is similar to weighted_vote but uses confidence as an additional multiplier.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        weights: Dictionary of agent_name -> weight

    Returns:
        Tuple of (is_phishing, confidence, voting_details)
    """
    phishing_score = 0.0
    legitimate_score = 0.0
    total_weight = 0.0

    votes = {}

    for agent_name, output in outputs.items():
        if output.error:
            continue

        weight = weights.get(agent_name, 1.0)
        total_weight += weight

        # Confidence-weighted score
        if output.is_phishing:
            score = weight * output.confidence
            phishing_score += score
            votes[agent_name] = {
                "vote": "phishing",
                "weight": weight,
                "confidence": output.confidence,
                "score": score
            }
        else:
            # For legitimate votes, use inverse confidence
            score = weight * (1.0 - output.confidence)
            legitimate_score += score
            votes[agent_name] = {
                "vote": "legitimate",
                "weight": weight,
                "confidence": output.confidence,
                "score": score
            }

    if total_weight == 0:
        return False, 0.0, {"error": "No valid agent outputs"}

    is_phishing = phishing_score > legitimate_score

    # Normalize confidence
    if is_phishing:
        confidence = phishing_score / (phishing_score + legitimate_score)
    else:
        confidence = legitimate_score / (phishing_score + legitimate_score)

    # Clamp confidence to [0, 1] (safety check)
    confidence = max(0.0, min(1.0, confidence))

    voting_details = {
        "method": "confidence_weighted",
        "total_weight": total_weight,
        "phishing_score": phishing_score,
        "legitimate_score": legitimate_score,
        "votes": votes,
        "winner": "phishing" if is_phishing else "legitimate"
    }

    return is_phishing, confidence, voting_details


def vote(
    outputs: Dict[str, AgentOutput],
    weights: Dict[str, float],
    method: str = "weighted"
) -> Tuple[bool, float, Dict]:
    """
    Perform voting using specified method.

    Args:
        outputs: Dictionary of agent_name -> AgentOutput
        weights: Dictionary of agent_name -> weight
        method: Voting method ("weighted", "majority", "confidence_weighted")

    Returns:
        Tuple of (is_phishing, confidence, voting_details)
    """
    if method == "weighted":
        return weighted_vote(outputs, weights)
    elif method == "majority":
        return majority_vote(outputs)
    elif method == "confidence_weighted":
        return confidence_weighted_vote(outputs, weights)
    else:
        raise ValueError(f"Unknown voting method: {method}")
