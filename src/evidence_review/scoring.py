"""Transparent transferability scoring for evidence relevant to Peruvian anchoveta."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TransferabilityAssessment(BaseModel):
    ecological_similarity: int = Field(ge=0, le=3)
    climate_relevance: int = Field(ge=0, le=3)
    management_relevance: int = Field(ge=0, le=2)
    evidence_strength: int = Field(ge=0, le=2)
    operational_specificity: int = Field(ge=0, le=2)
    data_feasibility_peru: int = Field(ge=0, le=2)
    total_score: int | None = Field(default=None, ge=0, le=14)
    transferability_class: str | None = None
    rationale: str = Field(min_length=10)
    limiting_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def calculate_score(self) -> "TransferabilityAssessment":
        calculated = (
            self.ecological_similarity
            + self.climate_relevance
            + self.management_relevance
            + self.evidence_strength
            + self.operational_specificity
            + self.data_feasibility_peru
        )
        if self.total_score is not None and self.total_score != calculated:
            raise ValueError("total_score does not match component scores")
        self.total_score = calculated
        self.transferability_class = classify_transferability(calculated)
        return self


def classify_transferability(total_score: int) -> str:
    """Classify a 0-14 score without replacing qualitative interpretation."""

    if total_score >= 12:
        return "very_high"
    if total_score >= 9:
        return "high"
    if total_score >= 5:
        return "moderate"
    return "low"


def score_transferability(
    ecological_similarity: int,
    climate_relevance: int,
    management_relevance: int,
    evidence_strength: int,
    operational_specificity: int,
    data_feasibility_peru: int,
    rationale: str,
    limiting_conditions: list[str] | None = None,
) -> TransferabilityAssessment:
    """Build a validated assessment from explicit component scores."""

    return TransferabilityAssessment(
        ecological_similarity=ecological_similarity,
        climate_relevance=climate_relevance,
        management_relevance=management_relevance,
        evidence_strength=evidence_strength,
        operational_specificity=operational_specificity,
        data_feasibility_peru=data_feasibility_peru,
        rationale=rationale,
        limiting_conditions=limiting_conditions or [],
    )
