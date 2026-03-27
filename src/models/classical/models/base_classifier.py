from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseClassifier:
    threshold: float = 0.5

    def predict_score(self, features: dict[str, float | int | bool]) -> float:
        total = 0.0
        for value in features.values():
            if isinstance(value, bool):
                total += 0.15 if value else 0.0
            else:
                total += min(float(value) / 10.0, 0.2)
        return max(0.0, min(total, 1.0))

    def predict(self, features: dict[str, float | int | bool]) -> int:
        return int(self.predict_score(features) >= self.threshold)

    def predict_proba(self, features: dict[str, float | int | bool]) -> list[float]:
        score = self.predict_score(features)
        return [1.0 - score, score]
