"""Prompt templates for screening, extraction, and appraisal of mixed evidence."""

from __future__ import annotations

import json


SYSTEM_RULES = """
You are conducting a systematic evidence map on climate change and marine fisheries.
Use only the supplied source text and metadata. Do not add external facts.
Distinguish what the source reports, proposes, projects, recommends, or legally requires.
Do not present a recommendation as an observed result, a project output as an outcome,
or an academic proposal as a binding rule.
Every finding must include an exact locator and a faithful evidence summary.
Use a short excerpt only when it directly supports the coded claim.
Return valid JSON only. Use null, false, empty lists, or 'unclear' when evidence is absent.
""".strip()


SCREENING_PROMPT = """
Determine whether this source should be included in a global evidence map on climate
change, environmental variability, adaptation, and marine fisheries.

Include only when all three conditions are true:
1. The source concerns marine fisheries, fishery governance, forage fish, small
   pelagics, or a directly relevant marine ecosystem.
2. It contains a climate, environmental-variability, extreme-event, precautionary,
   ecosystem, adaptation, or resilience element.
3. It contributes at least one codable finding, mechanism, method, projection,
   evaluation, recommendation, or implementation lesson.

Do not require the source to be a legal or official instrument. Scientific articles,
reports, chapters, projects, plans, protocols, and laws are all eligible, but their
source type and authority must be coded accurately.

Return a JSON object matching ScreeningDecision.

SOURCE METADATA:
{metadata}

SOURCE TEXT:
{source_text}
""".strip()


EXTRACTION_PROMPT = """
Extract exactly one traceable finding from this source.

First classify what the finding is: empirical result, model projection, legal
mechanism, management measure, policy commitment, project activity, project output,
project outcome, implementation barrier, implementation enabler, recommendation,
method, data resource, conceptual framework, or evidence gap.

Then identify the relevant evidence streams, climate drivers, biological responses,
fishery responses, governance mechanisms, management measures, method or design,
and relevance to small pelagics and Peruvian anchoveta.

Critical distinctions:
- What is proposed is not necessarily implemented.
- What is implemented is not necessarily effective.
- A project activity or output is not an outcome unless change is evaluated.
- A scenario result is not an observed trend.
- Correlation is not causation unless the design supports causal interpretation.
- A legal or policy statement is not operational unless responsibilities, triggers,
  procedures, or actions are identifiable.
- Indirect transferability must be labelled as indirect and justified.

Return a JSON object matching EvidenceFinding. Preserve the exact page, article,
section, table, figure, result, annex, or project component in unit_locator.

SOURCE METADATA:
{metadata}

SOURCE TEXT:
{source_text}
""".strip()


QUALITY_PROMPT = """
Appraise this source using only the criteria supplied for its source category.
Score each criterion as:
0 = absent or seriously inadequate
1 = partial or unclear
2 = adequate and traceable

Do not use journal prestige, institutional reputation, citation count, or legal force
as substitutes for methodological quality. Do not use methodological quality as a
substitute for legal authority. Explain every score briefly and identify critical
limitations that could alter interpretation or transferability.

APPRAISAL DOMAIN:
{appraisal_domain}

CRITERIA:
{criteria}

SOURCE METADATA:
{metadata}

SOURCE TEXT:
{source_text}
""".strip()


def _json(value: dict | list) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_screening_prompt(metadata: dict, source_text: str) -> str:
    return SCREENING_PROMPT.format(
        metadata=_json(metadata),
        source_text=source_text.strip(),
    )


def build_extraction_prompt(metadata: dict, source_text: str) -> str:
    return EXTRACTION_PROMPT.format(
        metadata=_json(metadata),
        source_text=source_text.strip(),
    )


def build_quality_prompt(
    metadata: dict,
    source_text: str,
    appraisal_domain: str,
    criteria: list[str],
) -> str:
    return QUALITY_PROMPT.format(
        appraisal_domain=appraisal_domain,
        criteria=_json(criteria),
        metadata=_json(metadata),
        source_text=source_text.strip(),
    )
