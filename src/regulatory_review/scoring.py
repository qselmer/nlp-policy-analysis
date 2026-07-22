"""Transparent transferability scoring for candidate regulatory mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TransferabilityComponents:
    ecological_similarity: int
    climate_operationalisation: int
    trigger_specificity: int
    response_predefinition: int
    legal_force: int
    data_feasibility_peru: int

    def validate(self) -> None:
        allowed = {
            "ecological_similarity": range(0, 4),
            "climate_operationalisation": range(0, 4),
            "trigger_specificity": range(0, 3),
            "response_predefinition": range(0, 3),
            "legal_force": range(0, 3),
            "data_feasibility_peru": range(0, 3),
        }
        for name, value in asdict(self).items():
            if value not in allowed[name]:
                raise ValueError(f"{name}={value} is outside the permitted range")

    @property
    def total(self) -> int:
        self.validate()
        return sum(asdict(self).values())

    @property
    def category(self) -> str:
        score = self.total
        if score <= 4:
            return "low"
        if score <= 8:
            return "moderate"
        if score <= 11:
            return "high"
        return "very_high"


def score_transferability(**kwargs: int) -> dict[str, int | str]:
    """Return component scores, total score, and interpretation category."""
    components = TransferabilityComponents(**kwargs)
    return {
        **asdict(components),
        "total": components.total,
        "category": components.category,
    }
