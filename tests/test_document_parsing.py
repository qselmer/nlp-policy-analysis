from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from evidence_review import document_parsing as dp


@pytest.fixture
def config():
    return {
        "paths": {"text_dir": "data/processed/text"},
        "validation": {
            "verify_pdf_signature": True,
            "verify_checksum": True,
            "strict_checksum": True,
        },
        "page_diagnostics": {
            "minimum_page_characters": 20,
            "scanned_document_threshold": 0.5,
            "minimum_document_characters": 30,
        },
        "text_cleaning": {
            "remove_null_bytes": True,
            "unicode_normalization": "NFKC",
            "dehyphenate_line_breaks": True,
            "preserve_paragraphs": True,
        },
        "section_detection": {
            "heading_max_characters": 120,
            "heading_max_words": 12,
            "heading_keywords": [
                "abstract",
                "introduction",
                "methods",
                "results",
                "discussion",
                "conclusions",
            ],
        },
        "chunking": {
            "target_characters": 80,
            "overlap_characters": 15,
            "minimum_chunk_characters": 20,
            "minimum_break_fraction": 0.5,
        },
    }


def test_clean_extracted_text_preserves_paragraphs_and_dehyphenates(config):
    raw = "Climate adap-\ntation\nrequires evidence.\n\nSecond\tparagraph.\x00"
    cleaned = dp.clean_extracted_text(raw, config["text_cleaning"])
    assert cleaned == "Climate adaptation requires evidence.\n\nSecond paragraph."


def test_classify_page_text():
    assert dp.classify_page_text("") == "empty"
    assert dp.classify_page_text("few", 10) == "low_text"
    assert dp.classify_page_text("long enough text", 10) == "text"


def test_detect_heading_is_conservative(config):
    settings = config["section_detection"]
    assert dp.detect_heading("2. Methods", settings) == (True, "numbered")
    assert dp.detect_heading("RESULTS", settings) == (True, "keyword")
    assert dp.detect_heading("This is a complete sentence.", settings) == (False, "")


def test_segment_sections_preserves_page_ranges(config):
    pages = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "include",
                "local_path": "paper.pdf",
                "source_checksum_sha256": "abc",
                "page_number": 1,
                "extraction_status": "text",
                "text": "Introduction\n\nOpening evidence paragraph with enough words.",
            },
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "include",
                "local_path": "paper.pdf",
                "source_checksum_sha256": "abc",
                "page_number": 2,
                "extraction_status": "text",
                "text": "Methods\n\nSampling and modelling details are described here.",
            },
        ]
    )
    sections, blocks = dp.segment_document_sections(pages, config)
    assert list(sections["section_heading"]) == ["Introduction", "Methods"]
    assert list(sections["start_page"]) == [1, 2]
    assert set(blocks["page_number"]) == {1, 2}


def test_chunking_creates_stable_page_traceable_chunks(config):
    blocks = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "include",
                "local_path": "paper.pdf",
                "source_checksum_sha256": "abc",
                "section_id": "sec_1",
                "section_index": 1,
                "section_heading": "Results",
                "heading_detection_method": "keyword",
                "page_number": 3,
                "block_index": 1,
                "text": "A" * 60,
            },
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "include",
                "local_path": "paper.pdf",
                "source_checksum_sha256": "abc",
                "section_id": "sec_1",
                "section_index": 1,
                "section_heading": "Results",
                "heading_detection_method": "keyword",
                "page_number": 4,
                "block_index": 2,
                "text": "B" * 60,
            },
        ]
    )
    first = dp.chunk_section_blocks(blocks, config)
    second = dp.chunk_section_blocks(blocks, config)
    assert len(first) >= 2
    assert first["chunk_id"].tolist() == second["chunk_id"].tolist()
    assert first["start_page"].min() == 3
    assert first["end_page"].max() == 4
    assert first["text"].str.len().gt(0).all()


def test_extract_manifest_documents_and_ocr_flag(tmp_path, config, monkeypatch):
    pdf = tmp_path / "src_1__paper.pdf"
    pdf.write_bytes(b"%PDF-fake test bytes")
    checksum = sha256(pdf.read_bytes()).hexdigest()

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        is_encrypted = False

        def __init__(self, path):
            assert Path(path) == pdf
            self.pages = [
                FakePage("Introduction\n\nA sufficiently long first page of extracted text."),
                FakePage(""),
            ]

    monkeypatch.setattr(dp, "PdfReader", FakeReader)
    manifest = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "uncertain",
                "local_path": pdf.name,
                "file_name": pdf.name,
                "checksum_sha256": checksum,
            }
        ]
    )
    inventory, pages, diagnostics, errors = dp.extract_manifest_documents(
        manifest,
        config,
        project_root=tmp_path,
    )
    assert errors.empty
    assert inventory.loc[0, "parsing_status"] == "parsed"
    assert inventory.loc[0, "checksum_matches"] == "true"
    assert inventory.loc[0, "page_count"] == 2
    assert inventory.loc[0, "ocr_recommended"] == "true"
    assert list(pages["extraction_status"]) == ["text", "empty"]
    assert diagnostics.loc[1, "possible_scanned_page"] == "true"


def test_strict_checksum_mismatch_stops_document(tmp_path, config, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake")

    class ShouldNotRun:
        def __init__(self, path):
            raise AssertionError("PdfReader must not run after checksum mismatch")

    monkeypatch.setattr(dp, "PdfReader", ShouldNotRun)
    manifest = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "title": "Paper",
                "local_path": pdf.name,
                "checksum_sha256": "wrong",
            }
        ]
    )
    inventory, pages, _, errors = dp.extract_manifest_documents(
        manifest,
        config,
        project_root=tmp_path,
    )
    assert inventory.loc[0, "parsing_status"] == "failed"
    assert pages.empty
    assert "checksum" in errors.loc[0, "error_message"]


def test_full_text_screening_corpus_marks_uncertain():
    inventory = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "title": "Paper",
                "screening_decision": "uncertain",
                "local_path": "paper.pdf",
                "observed_checksum_sha256": "abc",
                "page_count": 1,
                "total_characters": 20,
                "ocr_recommended": "false",
                "parsing_status": "parsed",
            }
        ]
    )
    pages = pd.DataFrame(
        [{"source_id": "src_1", "page_number": 1, "text": "Full text"}]
    )
    corpus = dp.build_full_text_screening_corpus(inventory, pages)
    assert corpus.loc[0, "full_text_screening_required"] == "true"
    assert "[[PAGE 1]]" in corpus.loc[0, "full_text"]


def test_validate_parsing_outputs_detects_duplicate_page():
    inventory = pd.DataFrame(
        [
            {
                "source_id": "src_1",
                "page_count": 2,
                "parsing_status": "parsed",
                "checksum_matches": "true",
            }
        ]
    )
    pages = pd.DataFrame(
        [
            {"source_id": "src_1", "page_number": 1},
            {"source_id": "src_1", "page_number": 1},
        ]
    )
    issues = dp.validate_parsing_outputs(
        inventory,
        pages,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert "duplicate_source_page" in set(issues["issue"])
