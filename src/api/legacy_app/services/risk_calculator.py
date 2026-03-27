"""
Risk calculation and aggregation utilities.
"""
from typing import Dict, Any, List
from src.api.legacy_app.schemas.enums import Verdict, RiskLevel


class RiskCalculator:
    """
    Calculate risk scores and aggregate model predictions.
    """

    @staticmethod
    def calculate_risk_score(confidence: float, verdict: Verdict) -> int:
        """
        Calculate risk score (0-100) from confidence and verdict.

        Args:
            confidence: Model confidence (0-1)
            verdict: Prediction verdict

        Returns:
            Risk score 0-100
        """
        if verdict == Verdict.PHISHING:
            # High confidence phishing = high risk
            return int(confidence * 100)
        elif verdict == Verdict.LEGITIMATE:
            # Inverse of confidence for legitimate
            return int((1 - confidence) * 30)  # Max 30 for legitimate
        else:  # SUSPICIOUS
            # Medium risk
            return int(50 + confidence * 30)

    @staticmethod
    def risk_score_to_level(risk_score: int) -> RiskLevel:
        """
        Convert risk score to risk level category.

        Args:
            risk_score: Risk score 0-100

        Returns:
            Risk level category
        """
        if risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    @staticmethod
    def aggregate_predictions(
        predictions: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple model predictions using weighted voting.

        Args:
            predictions: List of model predictions
            weights: Model weights (must sum to 1.0)

        Returns:
            Aggregated prediction
        """
        if not predictions:
            return {
                "verdict": Verdict.SUSPICIOUS,
                "confidence": 0.5,
                "risk_score": 50,
                "risk_level": RiskLevel.MEDIUM
            }

        # Calculate weighted scores
        phishing_score = 0.0
        legitimate_score = 0.0
        total_weight = 0.0

        for pred in predictions:
            model_name = pred.get("model_name", "unknown")
            weight = weights.get(model_name, 0)

            if weight == 0:
                continue

            verdict = pred.get("verdict")
            confidence = pred.get("confidence", 0.5)

            if verdict == Verdict.PHISHING:
                phishing_score += weight * confidence
            elif verdict == Verdict.LEGITIMATE:
                legitimate_score += weight * confidence
            else:  # SUSPICIOUS
                phishing_score += weight * confidence * 0.5
                legitimate_score += weight * confidence * 0.5

            total_weight += weight

        # Normalize
        if total_weight > 0:
            phishing_score /= total_weight
            legitimate_score /= total_weight

        # Determine verdict
        if phishing_score > legitimate_score + 0.2:
            verdict = Verdict.PHISHING
            confidence = phishing_score
        elif legitimate_score > phishing_score + 0.2:
            verdict = Verdict.LEGITIMATE
            confidence = legitimate_score
        else:
            verdict = Verdict.SUSPICIOUS
            confidence = max(phishing_score, legitimate_score)

        # Calculate risk score
        risk_score = RiskCalculator.calculate_risk_score(confidence, verdict)
        risk_level = RiskCalculator.risk_score_to_level(risk_score)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "phishing_score": phishing_score,
            "legitimate_score": legitimate_score,
            "models_used": len(predictions)
        }

    @staticmethod
    def generate_explanation(
        verdict: Verdict,
        confidence: float,
        risk_factors: List[str]
    ) -> str:
        """
        Generate human-readable explanation.

        Args:
            verdict: Prediction verdict
            confidence: Model confidence
            risk_factors: List of risk factors

        Returns:
            Explanation string
        """
        if verdict == Verdict.PHISHING:
            intro = f"This email exhibits strong indicators of phishing (confidence: {confidence:.1%})."
        elif verdict == Verdict.LEGITIMATE:
            intro = f"This email appears to be legitimate (confidence: {confidence:.1%})."
        else:
            intro = f"This email contains some suspicious elements (confidence: {confidence:.1%})."

        if risk_factors:
            factors = ". ".join(risk_factors[:3])  # Limit to top 3
            return f"{intro} {factors}."
        else:
            return intro
