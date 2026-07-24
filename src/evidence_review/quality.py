"""Quality-appraisal criteria selected by source type."""

from __future__ import annotations

from .schema import SourceType


COMMON_CRITERIA = [
    "source_authenticity",
    "evidence_traceability",
    "objective_clarity",
    "limitation_transparency",
    "uncertainty_treatment",
]

SCIENTIFIC_CRITERIA = [
    "design_appropriateness",
    "data_adequacy",
    "methodological_transparency",
    "validation_or_sensitivity",
    "reproducibility",
    "conclusion_support",
]

TECHNICAL_REPORT_CRITERIA = [
    "institutional_authority",
    "data_coverage",
    "method_transparency",
    "review_process",
    "indicator_clarity",
    "conclusion_support",
]

PROJECT_CRITERIA = [
    "theory_of_change",
    "baseline_quality",
    "indicator_quality",
    "evaluation_design",
    "outcome_evidence",
    "sustainability_and_scalability",
]

POLICY_AND_PLAN_CRITERIA = [
    "mandate_and_scope",
    "responsibility_assignment",
    "financing_and_capacity",
    "timeline_and_targets",
    "monitoring_and_review",
    "operational_mechanisms",
]

LEGAL_CRITERIA = [
    "official_version",
    "legal_force_clarity",
    "jurisdiction_and_scope",
    "authority_and_duties",
    "trigger_and_response_specificity",
    "implementation_evidence",
]


SCIENTIFIC_TYPES = {
    SourceType.SCIENTIFIC_ARTICLE,
    SourceType.SYSTEMATIC_REVIEW,
    SourceType.BOOK_CHAPTER,
    SourceType.BOOK,
    SourceType.THESIS_OR_DISSERTATION,
    SourceType.CONFERENCE_PAPER,
    SourceType.DATASET_OR_DATA_PAPER,
}

TECHNICAL_TYPES = {
    SourceType.SCIENTIFIC_ADVICE,
    SourceType.STOCK_ASSESSMENT,
    SourceType.TECHNICAL_REPORT,
    SourceType.INSTITUTIONAL_REPORT,
    SourceType.TECHNICAL_PROTOCOL,
    SourceType.GUIDELINE_OR_MANUAL,
}

PROJECT_TYPES = {
    SourceType.PROJECT_DOCUMENT,
    SourceType.PROJECT_EVALUATION,
}

POLICY_TYPES = {
    SourceType.POLICY,
    SourceType.STRATEGY,
    SourceType.ADAPTATION_PLAN,
    SourceType.FISHERY_MANAGEMENT_PLAN,
    SourceType.HARVEST_STRATEGY,
}

LEGAL_TYPES = {
    SourceType.INTERNATIONAL_AGREEMENT,
    SourceType.REGIONAL_MEASURE,
    SourceType.NATIONAL_LAW,
    SourceType.REGULATION,
}


def appraisal_domain(source_type: SourceType) -> str:
    """Return the most appropriate appraisal domain for a source type."""

    if source_type in SCIENTIFIC_TYPES:
        return "scientific"
    if source_type in TECHNICAL_TYPES:
        return "technical_report"
    if source_type in PROJECT_TYPES:
        return "project"
    if source_type in POLICY_TYPES:
        return "policy_and_plan"
    if source_type in LEGAL_TYPES:
        return "legal"
    return "technical_report"


def appraisal_criteria(source_type: SourceType, include_common: bool = True) -> list[str]:
    """Return criteria without using source prestige as a quality proxy."""

    domain = appraisal_domain(source_type)
    domain_criteria = {
        "scientific": SCIENTIFIC_CRITERIA,
        "technical_report": TECHNICAL_REPORT_CRITERIA,
        "project": PROJECT_CRITERIA,
        "policy_and_plan": POLICY_AND_PLAN_CRITERIA,
        "legal": LEGAL_CRITERIA,
    }[domain]

    if include_common:
        return [*COMMON_CRITERIA, *domain_criteria]
    return list(domain_criteria)


def quality_rating(total_score: int, maximum_score: int) -> str:
    """Translate an appraisal proportion into a transparent descriptive class."""

    if maximum_score <= 0:
        return "not_applicable"
    proportion = total_score / maximum_score
    if proportion >= 0.75:
        return "high"
    if proportion >= 0.50:
        return "moderate"
    return "low"
