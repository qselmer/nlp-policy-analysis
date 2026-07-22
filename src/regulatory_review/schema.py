"""Data models for the fisheries climate-regulation evidence map."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class LegalStatus(str, Enum):
    BINDING = "binding"
    PARTIALLY_BINDING = "partially_binding"
    NON_BINDING = "non_binding"
    UNCLEAR = "unclear"


class ResponseType(str, Enum):
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    DISCRETIONARY = "discretionary"
    ADVISORY = "advisory"
    NOT_SPECIFIED = "not_specified"


class RelevanceLevel(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    NONE = "none"
    UNCLEAR = "unclear"


class RegulatoryRecord(BaseModel):
    """One evidence-backed regulatory mechanism extracted from a source."""

    record_id: str = Field(min_length=3)
    instrument_name: str = Field(min_length=2)
    jurisdiction: str = Field(min_length=2)
    region: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    version_date: str | None = None
    instrument_type: str
    legal_status: LegalStatus = LegalStatus.UNCLEAR
    competent_authority: str | None = None
    official_source_url: HttpUrl
    source_language: str | None = None

    article_or_section: str = Field(
        description="Exact article, section, annex, or page supporting the record."
    )
    supporting_excerpt: str = Field(
        min_length=10,
        description="Short verbatim excerpt supporting the coded claims.",
    )

    fishery_scope: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    stock_or_management_unit: str | None = None
    fleet_scope: list[str] = Field(default_factory=list)
    small_pelagic_relevance: RelevanceLevel = RelevanceLevel.UNCLEAR

    climate_change_explicit: bool = False
    environmental_variability_explicit: bool = False
    climate_integration_level: int = Field(default=0, ge=0, le=4)
    principles: list[str] = Field(default_factory=list)

    management_measures: list[str] = Field(default_factory=list)
    trigger_types: list[str] = Field(default_factory=list)
    trigger_description: str | None = None
    quantitative_threshold_defined: bool = False
    threshold_value: str | None = None
    management_response: list[str] = Field(default_factory=list)
    response_type: ResponseType = ResponseType.NOT_SPECIFIED
    quota_can_be_revised_in_season: bool | None = None

    scientific_body: str | None = None
    scientific_advice_binding: Literal["yes", "no", "partial", "unclear"] = "unclear"
    review_frequency: str | None = None

    relevance_to_anchoveta: str | None = None
    limitations_for_transfer: str | None = None
    transferability_score: int | None = Field(default=None, ge=0, le=14)
    coder_confidence: float = Field(default=0.5, ge=0, le=1)
    human_validation_status: Literal[
        "not_reviewed", "accepted", "corrected", "rejected"
    ] = "not_reviewed"
    uncertainty_notes: str | None = None

    @model_validator(mode="after")
    def check_evidence_consistency(self) -> "RegulatoryRecord":
        if self.quantitative_threshold_defined and not self.threshold_value:
            raise ValueError(
                "threshold_value is required when quantitative_threshold_defined is true"
            )
        if self.climate_integration_level >= 3 and not (
            self.climate_change_explicit or self.environmental_variability_explicit
        ):
            raise ValueError(
                "Climate integration level 3-4 requires explicit climate or environmental evidence"
            )
        return self


class ScreeningDecision(BaseModel):
    document_id: str
    include: bool
    exclusion_reason: str | None = None
    fisheries_related: bool
    climate_or_adaptive_element: bool
    official_instrument: bool
    priority_group: list[str] = Field(default_factory=list)
    reviewer_notes: str | None = None
    human_validation_status: Literal[
        "not_reviewed", "accepted", "corrected", "rejected"
    ] = "not_reviewed"

    @model_validator(mode="after")
    def check_exclusion_reason(self) -> "ScreeningDecision":
        if not self.include and not self.exclusion_reason:
            raise ValueError("An exclusion_reason is required for excluded documents")
        return self
