"""Prompt templates for screening and structured regulatory extraction."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


SYSTEM_RULES = """
You are analysing official fisheries and climate-policy instruments.
Use only the supplied source text. Do not use external knowledge.
Do not infer that a legal mechanism exists when the text is silent.
Every substantive claim must be supported by an exact article/section and excerpt.
Return valid JSON only. Use null, false, or 'unclear' when evidence is absent.
Distinguish a declarative climate reference from an operational rule linked to a decision.
""".strip()


SCREENING_PROMPT = """
Determine whether this document should be included in a global evidence map of
fisheries regulation under environmental variability and climate change.

Include only when all three conditions are met:
1. It is an official legal, regulatory, policy, strategy, or management instrument.
2. It applies to marine fisheries, fishery governance, or a relevant marine ecosystem.
3. It contains at least one climate, environmental-variability, precautionary,
   ecosystem, or adaptive-management element.

Return these fields:
- document_id
- include
- exclusion_reason
- fisheries_related
- climate_or_adaptive_element
- official_instrument
- priority_group
- reviewer_notes
- human_validation_status = 'not_reviewed'

DOCUMENT METADATA:
{metadata}

SOURCE TEXT:
{source_text}
""".strip()


EXTRACTION_PROMPT = """
Extract one evidence-backed regulatory mechanism from the source text.
Focus on climate change, environmental variability, precaution, ecosystem-based
management, adaptive management, harvest control rules, closures, quota revision,
scientific authority, legal triggers, and review requirements.

Important distinctions:
- A general aspiration is not an operational mechanism.
- Scientific advice is not legally binding unless the text says so.
- A management response is not automatic unless the text establishes a direct rule.
- Climate integration level:
  0 absent
  1 declarative mention
  2 mandatory risk assessment
  3 environmental indicators used in advice
  4 indicators linked to operational decisions

Return a JSON object matching the RegulatoryRecord schema.

DOCUMENT METADATA:
{metadata}

SOURCE TEXT:
{source_text}
""".strip()


def load_taxonomy(path: str | Path = "config/taxonomy.yml") -> dict:
    """Load the controlled vocabulary used by the project."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_screening_prompt(metadata: dict, source_text: str) -> str:
    return SCREENING_PROMPT.format(
        metadata=json.dumps(metadata, ensure_ascii=False, indent=2),
        source_text=source_text.strip(),
    )


def build_extraction_prompt(metadata: dict, source_text: str) -> str:
    return EXTRACTION_PROMPT.format(
        metadata=json.dumps(metadata, ensure_ascii=False, indent=2),
        source_text=source_text.strip(),
    )
