from __future__ import annotations

from src.explainability.cognitive_guidelines import build_guidelines
from src.explainability.domain_mapper import map_features_to_concepts
from src.explainability.shap_extractor import extract_shap_like_contributions


def generate_explanation(prediction: dict) -> str:
    features = prediction.get("features", {})
    top_features = extract_shap_like_contributions(features)
    concepts = map_features_to_concepts(top_features)
    guidance = build_guidelines(str(prediction.get("label", "unknown")), concepts)
    return (
        f"Prediction: {prediction.get('label', 'unknown')} with confidence {prediction.get('confidence', 0):.2f}. "
        f"Top evidence areas were {', '.join(concepts[:3]) or 'limited heuristics'}. "
        f"Recommended analyst actions: {' '.join(guidance)}"
    )
