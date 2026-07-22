from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_review.full_text import (
    build_full_text_manifest,
    build_retrieval_queue,
    classify_retrieval_status,
    download_open_access_pdf,
    initialise_retrieval_sheet,
    load_full_text_config,
    parse_openalex_work,
    register_local_files,
    retrieval_summary,
    split_retrieval_outputs,
    validate_retrieval_sheet,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_full_text_config(ROOT / "config" / "full_text_retrieval.yml")


def _screening_frame(source_id: str, title: str, doi: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": source_id,
                "title": title,
                "abstract": "Climate and fisheries evidence.",
                "authors_or_organisation": "Example",
                "publisher": "Example Journal",
                "year": "2024",
                "source_type": "scientific_article",
                "source_status": "peer_reviewed",
                "source_language": "en",
                "doi": doi,
                "primary_url": "https://example.org/landing",
                "landing_page_url": "",
            }
        ]
    )


def test_build_queue_combines_include_and_uncertain():
    included = _screening_frame(
        "src_inc",
        "Included",
        "https://doi.org/10.1/inc",
    )
    uncertain = _screening_frame("src_unc", "Uncertain")

    queue = build_retrieval_queue(included, uncertain)

    assert len(queue) == 2
    assert set(queue["screening_decision"]) == {"include", "uncertain"}
    uncertain_required = queue.loc[
        queue["source_id"] == "src_unc",
        "full_text_screening_required",
    ].iloc[0]
    assert uncertain_required == "true"
    assert queue.loc[queue["source_id"] == "src_inc", "doi"].iloc[0] == "10.1/inc"


def test_initialise_preserves_retrieval_fields():
    queue = build_retrieval_queue(
        _screening_frame("src_1", "A"),
        pd.DataFrame(),
    )
    existing = queue.copy()
    existing.loc[0, "retrieval_status"] = "landing_page_only"
    existing.loc[0, "retrieval_notes"] = "checked"

    refreshed = initialise_retrieval_sheet(queue, existing)

    assert refreshed.loc[0, "retrieval_status"] == "landing_page_only"
    assert refreshed.loc[0, "retrieval_notes"] == "checked"


def test_parse_openalex_and_classify_oa_pdf():
    payload = {
        "id": "https://openalex.org/W1",
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "best_oa_location": {
            "is_oa": True,
            "landing_page_url": "https://example.org/article",
            "pdf_url": "https://example.org/article.pdf",
            "license": "cc-by",
            "version": "publishedVersion",
            "source": {"display_name": "Journal"},
        },
    }

    fields = parse_openalex_work(payload)
    status = classify_retrieval_status(fields)

    assert fields["openalex_id"].endswith("/W1")
    assert fields["is_open_access"] == "true"
    assert fields["best_pdf_url"].endswith(".pdf")
    assert status == "available_open_access"


def test_landing_page_is_not_full_text():
    status = classify_retrieval_status(
        {
            "primary_url": "https://example.org/landing",
            "best_pdf_url": "",
            "is_open_access": "",
        }
    )
    assert status == "landing_page_only"


def test_register_local_pdf_and_manifest(tmp_path: Path):
    queue = build_retrieval_queue(
        _screening_frame("src_local", "Local PDF"),
        pd.DataFrame(),
    )
    pdf = tmp_path / "src_local__local-pdf.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexample")

    registered, errors = register_local_files(
        queue,
        tmp_path,
        project_root=tmp_path,
    )
    manifest = build_full_text_manifest(registered)

    assert errors.empty
    assert registered.loc[0, "retrieval_status"] == "available_local"
    assert registered.loc[0, "pdf_signature_valid"] == "true"
    assert len(registered.loc[0, "checksum_sha256"]) == 64
    assert len(manifest) == 1


def test_validate_available_local_requires_path():
    queue = build_retrieval_queue(
        _screening_frame("src_bad", "Bad"),
        pd.DataFrame(),
    )
    queue.loc[0, "retrieval_status"] = "available_local"

    issues = validate_retrieval_sheet(queue, CONFIG)

    assert "available_local_requires_local_path" in set(issues["issue"])


class _FakeDownloadResponse:
    status_code = 200
    headers = {"Content-Type": "application/pdf"}

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield b"%PDF-1.7\n"
        yield b"content"


class _FakeSession:
    def get(self, *args, **kwargs):
        del args, kwargs
        return _FakeDownloadResponse()


def test_download_open_access_pdf(tmp_path: Path):
    row = {
        "source_id": "src_oa",
        "title": "Open access example",
        "retrieval_status": "available_open_access",
        "is_open_access": "true",
        "best_pdf_url": "https://example.org/open.pdf",
    }

    result = download_open_access_pdf(
        row,
        tmp_path,
        config=CONFIG,
        session=_FakeSession(),
    )

    assert result["retrieval_status"] == "available_local"
    assert result["pdf_signature_valid"] == "true"
    assert (tmp_path / result["file_name"]).exists()


def test_download_refuses_unconfirmed_access(tmp_path: Path):
    row = {
        "source_id": "src_closed",
        "title": "Closed",
        "retrieval_status": "restricted_access",
        "is_open_access": "false",
        "best_pdf_url": "https://example.org/closed.pdf",
    }

    with pytest.raises(ValueError):
        download_open_access_pdf(
            row,
            tmp_path,
            config=CONFIG,
            session=_FakeSession(),
        )


def test_split_outputs_and_summary():
    sheet = build_retrieval_queue(
        _screening_frame("src_a", "A"),
        _screening_frame("src_b", "B"),
    )
    sheet.loc[0, "retrieval_status"] = "available_open_access"
    sheet.loc[0, "best_pdf_url"] = "https://example.org/a.pdf"
    sheet.loc[0, "is_open_access"] = "true"
    sheet.loc[1, "retrieval_status"] = "not_found"

    groups = split_retrieval_outputs(sheet)
    summary = retrieval_summary(sheet)

    assert len(groups["available"]) == 1
    assert len(groups["unavailable"]) == 1
    assert len(groups["screening_queue"]) == 1
    queue_total = summary.loc[
        summary["metric"] == "queue_total",
        "value",
    ].iloc[0]
    assert int(queue_total) == 2
