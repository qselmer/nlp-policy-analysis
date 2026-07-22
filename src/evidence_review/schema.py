"""Pydantic models for a multitype climate-fisheries evidence map."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class SourceType(str, Enum):
    INTERNATIONAL_AGREEMENT = "international_agreement"
    REGIONAL_MEASURE = "regional_measure"
    NATIONAL_LAW = "national_law"
    REGULATION = "regulation"
    POLICY = "policy"
    STRATEGY = "strategy"
    ADAPTATION_PLAN = "adaptation_plan"
    FISHERY_MANAGEMENT_PLAN = "fishery_management_plan"
    HARVEST_STRATEGY = "harvest_strategy"
    TECHNICAL_PROTOCOL = "technical_protocol"
    SCIENTIFIC_ADVICE = "scientific_advice"
    STOCK_ASSESSMENT = "stock_assessment"
    TECHNICAL_REPORT = "technical_report"
    INSTITUTIONAL_REPORT = "institutional_report"
    PROJECT_DOCUMENT = "project_document"
    PROJECT_EVALUATION = "project_evaluation"
    GUIDELINE_OR_MANUAL = "guideline_or_manual"
    SCIENTIFIC_ARTICLE = "scientific_article"
    SYSTEMATIC_REVIEW = "systematic_review"
    BOOK_CHAPTER = "book_chapter"
    BOOK = "book"
    THESIS_OR_DISSERTATION = "thesis_or_dissertation"
    CONFERENCE_PAPER = "conference_paper"
    DATASET_OR_DATA_PAPER = "dataset_or_data_paper"
    OTHER = "other"


class SourceStatus(str, Enum):
    OFFICIAL_BINDING = "official_binding"
    OFFICIAL_NON_BINDING = "official_non_binding"
    OFFICIAL_TECHNICAL = "official_technical"
    PEER_REVIEWED = "peer_reviewed"
    EDITORIALLY_REVIEWED = "editorially_reviewed"
    INSTITUTIONAL_GREY_LITERATURE = "institutional_grey_literature"
    PROJECT_OUTPUT = "project_output"
    PREPRINT = "preprint"
    UNPUBLISHED = "unpublished"
    UNCLEAR = "unclear"


class RelevanceLevel(str, Enum):
    DIRECT = "direct"
    INDIRECT_HIGH = "indirect_high"
    INDIRECT_MODERATE = "indirect_moderate"
    INDIRECT_LOW = "indirect_low"
    NONE = "none"
    UNCLEAR = "unclear"


class ValidationStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class SourceRecord(BaseModel):
    """Bibliographic and provenance information for one source version."""

    source_id: str = Field(min_length=3)
    title: str = Field(min_length=2)
    source_type: SourceType
    source_subtype: str | None = None
    authors_or_organisation: list[str] = Field(default_factory=list)
    publisher: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    version_date: str | None = None
    jurisdiction: str | None = None
    geographic_scope: list[str] = Field(default_factory=list)
    fishery_scope: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    source_status: SourceStatus = SourceStatus.UNCLEAR
    legal_status: Literal["binding", "partially_binding", "non_binding", "unclear", "not_applicable"] = "not_applicable"
    doi: str | None = None
    primary_url: HttpUrl
    landing_page_url: HttpUrl | None = None
    access_date: str | None = None
    source_language: str | None = None
    local_path: str | None = None
    checksum_sha256: str | None = None
    supersedes_source_id: str | None = None
    notes: str | None = None


class ScreeningDecision(BaseModel):
    """First-pass eligibility decision for any supported source type."""

    source_id: str
    include: bool
    source_type: SourceType
    fisheries_or_marine_relevant: bool
    climate_environment_or_adaptation_element: bool
    contributes_codable_evidence: bool
    exclusion_reason: str | None = None
    priority_groups: list[str] = Field(default_factory=list)
    screening_stage: Literal["title_abstract", "executive_summary", "full_text"]
    reviewer_notes: str | None = None
    human_validation_status: ValidationStatus = ValidationStatus.NOT_REVIEWED

    @model_validator(mode="after")
    def validate_decision(self) -> "ScreeningDecision":
        expected_include = (
            self.fisheries_or_marine_relevant
            and self.climate_environment_or_adaptation_element
            and self.contributes_codable_evidence
        )
        if self.include != expected_include:
            raise ValueError(
                "include must equal the conjunction of the three eligibility criteria"
            )
        if not self.include and not self.exclusion_reason:
            raise ValueError("exclusion_reason is required for excluded sources")
        return self


class EvidenceFinding(BaseModel):
    """One traceable finding extracted from a source.

    A source can generate multiple findings. Each finding must point to a precise
    location such as an article, section, page, table, figure, result, or project
    component.
    """

    finding_id: str = Field(min_length=3)
    source_id: str = Field(min_length=3)
    source_type: SourceType
    finding_type: str
    evidence_streams: list[str] = Field(min_length=1)

    unit_locator: str = Field(
        min_length=1,
        description="Article, section, page, table, figure, result, annex, or project component.",
    )
    supporting_excerpt: str | None = Field(default=None, max_length=1500)
    evidence_summary: str = Field(
        min_length=10,
        description="Faithful synthesis of what the located source unit supports.",
    )

    geographic_scope: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    ecosystem_or_region: list[str] = Field(default_factory=list)
    fishery_scope: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    stock_or_management_unit: str | None = None

    climate_drivers: list[str] = Field(default_factory=list)
    environmental_variables: list[str] = Field(default_factory=list)
    biological_responses: list[str] = Field(default_factory=list)
    fishery_responses: list[str] = Field(default_factory=list)
    governance_mechanisms: list[str] = Field(default_factory=list)
    management_measures: list[str] = Field(default_factory=list)

    method_or_design: list[str] = Field(default_factory=list)
    study_period: str | None = None
    temporal_scale: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    sample_or_data_description: str | None = None
    quantitative_result: str | None = None
    result_direction: Literal[
        "positive", "negative", "mixed", "null", "non_directional", "unclear"
    ] = "non_directional"
    uncertainty_reported: bool | None = None

    relevance_to_small_pelagics: RelevanceLevel = RelevanceLevel.UNCLEAR
    relevance_to_anchoveta: RelevanceLevel = RelevanceLevel.UNCLEAR
    transferability_rationale: str | None = None
    limitations: list[str] = Field(default_factory=list)

    coder_confidence: float = Field(default=0.5, ge=0, le=1)
    human_validation_status: ValidationStatus = ValidationStatus.NOT_REVIEWED
    validation_notes: str | None = None

    @model_validator(mode="after")
    def check_traceability(self) -> "EvidenceFinding":
        if not self.supporting_excerpt and len(self.evidence_summary.strip()) < 20:
            raise ValueError(
                "A finding without an excerpt requires an evidence_summary of at least 20 characters"
            )
        if self.quantitative_result and not self.unit_locator:
            raise ValueError("Quantitative results require an exact source locator")
        return self


class QualityAppraisal(BaseModel):
    """Source-type-specific appraisal; scores are 0, 1, or 2."""

    appraisal_id: str
    source_id: str
    appraisal_domain: Literal[
        "scientific", "technical_report", "project", "policy_and_plan", "legal"
    ]
    criteria_scores: dict[str, int]
    criteria_notes: dict[str, str] = Field(default_factory=dict)
    total_score: int | None = Field(default=None, ge=0)
    maximum_score: int | None = Field(default=None, ge=0)
    overall_rating: Literal["low", "moderate", "high", "not_applicable"]
    critical_limitations: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    human_validation_status: ValidationStatus = ValidationStatus.NOT_REVIEWED

    @model_validator(mode="after")
    def validate_scores(self) -> "QualityAppraisal":
        invalid = {key: value for key, value in self.criteria_scores.items() if value not in {0, 1, 2}}
        if invalid:
            raise ValueError(f"Quality scores must be 0, 1, or 2: {invalid}")
        calculated_total = sum(self.criteria_scores.values())
        calculated_maximum = 2 * len(self.criteria_scores)
        if self.total_score is not None and self.total_score != calculated_total:
            raise ValueError("total_score does not match criteria_scores")
        if self.maximum_score is not None and self.maximum_score != calculated_maximum:
            raise ValueError("maximum_score does not match criteria_scores")
        self.total_score = calculated_total
        self.maximum_score = calculated_maximum
        return self
