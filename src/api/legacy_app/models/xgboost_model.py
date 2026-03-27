class XGBoostModel:
    def predict_proba(self, features: dict) -> list[float]:
        score = min(sum(float(v) for v in features.values() if not isinstance(v, bool)) / 100.0, 1.0)
        return [1.0 - score, score]
