"""Traceable PDF extraction, diagnostics, section detection, and chunking."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
import re
import unicodedata
from typing import Mapping

import pandas as pd
from pypdf import PdfReader
import yaml


INVENTORY_COLUMNS = [
    "source_id", "title", "screening_decision", "local_path", "file_name",
    "manifest_checksum_sha256", "observed_checksum_sha256", "checksum_matches",
    "pdf_signature_valid", "file_size_bytes", "page_count", "pages_with_text",
    "pages_low_text", "pages_empty", "pages_error", "total_characters",
    "total_words", "text_page_fraction", "low_or_empty_fraction",
    "ocr_recommended", "parsing_status", "parsed_at", "text_path", "notes",
]
PAGE_TEXT_COLUMNS = [
    "source_id", "title", "screening_decision", "local_path",
    "source_checksum_sha256", "page_number", "extraction_status",
    "character_count", "word_count", "text",
]
PAGE_DIAGNOSTIC_COLUMNS = [
    "source_id", "page_number", "extraction_status", "character_count",
    "word_count", "possible_scanned_page", "error_type", "error_message",
]
SECTION_COLUMNS = [
    "section_id", "source_id", "title", "screening_decision", "local_path",
    "source_checksum_sha256", "section_index", "section_heading",
    "heading_detection_method", "start_page", "end_page", "character_count",
    "word_count", "text",
]
BLOCK_COLUMNS = [
    "source_id", "title", "screening_decision", "local_path",
    "source_checksum_sha256", "section_id", "section_index", "section_heading",
    "heading_detection_method", "page_number", "block_index", "text",
]
CHUNK_COLUMNS = [
    "chunk_id", "source_id", "title", "screening_decision", "local_path",
    "source_checksum_sha256", "section_id", "section_index", "section_heading",
    "chunk_index", "start_page", "end_page", "character_start",
    "character_end", "character_count", "word_count", "text",
]
ERROR_COLUMNS = [
    "source_id", "stage", "page_number", "error_type", "error_message",
    "local_path",
]
ISSUE_COLUMNS = ["source_id", "output", "row_number", "field", "issue"]
OCR_COLUMNS = [
    "source_id", "title", "screening_decision", "local_path", "page_count",
    "pages_with_text", "pages_low_text", "pages_empty", "pages_error",
    "total_characters", "low_or_empty_fraction", "ocr_recommended",
    "parsing_status", "notes",
]
SCREENING_CORPUS_COLUMNS = [
    "source_id", "title", "screening_decision", "full_text_screening_required",
    "local_path", "source_checksum_sha256", "page_count", "total_characters",
    "ocr_recommended", "parsing_status", "full_text",
]


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def _stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(_clean(value) for value in values)
    return f"{prefix}_{sha1(payload.encode('utf-8')).hexdigest()[:14]}"


def load_document_parsing_config(path: str | Path = "config/document_parsing.yml") -> dict:
    """Load and validate document-parsing settings."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "paths", "validation", "page_diagnostics", "text_cleaning",
        "section_detection", "chunking",
    }
    if not isinstance(config, dict):
        raise ValueError("document parsing configuration must be a YAML mapping")
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing document parsing configuration sections: {missing}")
    target = int(config["chunking"]["target_characters"])
    overlap = int(config["chunking"]["overlap_characters"])
    minimum = int(config["chunking"]["minimum_chunk_characters"])
    threshold = float(config["page_diagnostics"]["scanned_document_threshold"])
    if target <= 0 or minimum <= 0 or overlap < 0 or overlap >= target:
        raise ValueError("invalid chunking configuration")
    if not 0 <= threshold <= 1:
        raise ValueError("scanned_document_threshold must be between 0 and 1")
    return config


def read_manifest_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read CSV using common UTF-8 and Windows encodings."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path, encoding=encoding, dtype=str, keep_default_na=False
            ), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Unable to decode manifest CSV: " + " | ".join(errors))


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_pdf_signature(path: str | Path) -> bool:
    with Path(path).open("rb") as stream:
        return stream.read(5) == b"%PDF-"


def resolve_local_path(local_path: object, project_root: str | Path) -> Path:
    value = _clean(local_path)
    if not value:
        raise ValueError("local_path is blank")
    path = Path(value)
    return (path if path.is_absolute() else Path(project_root) / path).resolve()


def clean_extracted_text(text: object, config: Mapping[str, object] | None = None) -> str:
    """Normalize extraction artefacts while preserving paragraph boundaries."""
    value = "" if text is None else str(text)
    settings = dict(config or {})
    if settings.get("remove_null_bytes", True):
        value = value.replace("\x00", "")
    normalization = _clean(settings.get("unicode_normalization", "NFKC"))
    if normalization:
        value = unicodedata.normalize(normalization, value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if settings.get("dehyphenate_line_breaks", True):
        value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    if not settings.get("preserve_paragraphs", True):
        return re.sub(r"\s+", " ", value).strip()
    paragraphs = [
        re.sub(r"[ \t\f\v\n]+", " ", item).strip()
        for item in re.split(r"\n\s*\n+", value)
    ]
    return "\n\n".join(item for item in paragraphs if item)


def classify_page_text(text: object, minimum_characters: int = 80) -> str:
    value = _clean(text)
    if not value:
        return "empty"
    return "low_text" if len(value) < int(minimum_characters) else "text"


def _normalise_heading(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(
        r"^\s*(?:(?:\d+(?:\.\d+)*)|(?:[ivxlcdm]+)|(?:[a-z]))"
        r"(?:[.)]|\s+-|\s+)\s*",
        "",
        value,
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def detect_heading(line: object, config: Mapping[str, object]) -> tuple[bool, str]:
    """Conservatively identify a likely section heading."""
    value = re.sub(r"\s+", " ", _clean(line))
    words = value.split()
    if (
        len(value) < 3
        or len(value) > int(config["heading_max_characters"])
        or len(words) > int(config["heading_max_words"])
        or value.endswith((".", ";", ",", "?", "!"))
    ):
        return False, ""
    numbered = re.match(
        r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])(?:[.)]|\s+-|\s+)", value
    )
    if numbered:
        return True, "numbered"
    keywords = {
        _normalise_heading(item)
        for item in config.get("heading_keywords", [])
        if _clean(item)
    }
    if _normalise_heading(value) in keywords:
        return True, "keyword"
    letters = [char for char in value if char.isalpha()]
    if letters and value.upper() == value and len(words) >= 2:
        return True, "uppercase"
    title_words = [word for word in words if any(char.isalpha() for char in word)]
    if len(title_words) >= 2 and len(value) <= 100:
        capitalized = sum(word[:1].isupper() for word in title_words)
        if capitalized / len(title_words) >= 0.8:
            return True, "title_case"
    return False, ""


def _metadata(row: Mapping[str, object]) -> dict[str, str]:
    return {
        "source_id": _clean(row.get("source_id")),
        "title": _clean(row.get("title")),
        "screening_decision": _clean(row.get("screening_decision")),
        "local_path": _clean(row.get("local_path")),
        "file_name": _clean(row.get("file_name")),
        "manifest_checksum_sha256": _clean(row.get("checksum_sha256")),
    }


def _inventory_row(meta: Mapping[str, str]) -> dict[str, object]:
    return {
        **meta,
        "observed_checksum_sha256": "", "checksum_matches": "",
        "pdf_signature_valid": "", "file_size_bytes": "", "page_count": 0,
        "pages_with_text": 0, "pages_low_text": 0, "pages_empty": 0,
        "pages_error": 0, "total_characters": 0, "total_words": 0,
        "text_page_fraction": 0.0, "low_or_empty_fraction": 0.0,
        "ocr_recommended": "false", "parsing_status": "not_started",
        "parsed_at": datetime.now(timezone.utc).isoformat(), "text_path": "",
        "notes": "",
    }


def _error(meta, stage, exc, page_number="") -> dict[str, object]:
    return {
        "source_id": meta["source_id"], "stage": stage,
        "page_number": page_number, "error_type": exc.__class__.__name__,
        "error_message": str(exc), "local_path": meta["local_path"],
    }


def extract_manifest_documents(
    manifest: pd.DataFrame,
    config: Mapping[str, object],
    *,
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract all registered local PDFs page by page."""
    required = {"source_id", "local_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    inventories, pages, diagnostics, errors = [], [], [], []
    page_limit = int(config["page_diagnostics"]["minimum_page_characters"])
    scan_limit = float(config["page_diagnostics"]["scanned_document_threshold"])
    document_limit = int(config["page_diagnostics"]["minimum_document_characters"])
    validation = config["validation"]

    for _, record in manifest.fillna("").iterrows():
        meta = _metadata(record)
        inventory = _inventory_row(meta)
        if not meta["source_id"]:
            exc = ValueError("source_id is blank")
            inventory.update(parsing_status="invalid_manifest", notes=str(exc))
            errors.append(_error(meta, "manifest_validation", exc))
            inventories.append(inventory)
            continue
        local_pages = []
        try:
            path = resolve_local_path(meta["local_path"], project_root)
            if not path.exists():
                raise FileNotFoundError(str(path))
            if not path.is_file():
                raise ValueError(f"local_path is not a file: {path}")
            checksum = file_sha256(path)
            expected = meta["manifest_checksum_sha256"].casefold()
            matches = bool(expected) and expected == checksum.casefold()
            signature = has_pdf_signature(path)
            inventory.update(
                file_name=meta["file_name"] or path.name,
                file_size_bytes=path.stat().st_size,
                observed_checksum_sha256=checksum,
                checksum_matches=_bool(matches) if expected else "",
                pdf_signature_valid=_bool(signature),
            )
            if validation.get("verify_pdf_signature", True) and not signature:
                raise ValueError("file does not have a PDF signature")
            if (
                validation.get("verify_checksum", True)
                and expected
                and not matches
                and validation.get("strict_checksum", True)
            ):
                raise ValueError("observed checksum differs from manifest checksum")

            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False) and not reader.decrypt(""):
                raise ValueError("encrypted PDF requires a password")
            inventory["page_count"] = len(reader.pages)

            for number, page in enumerate(reader.pages, start=1):
                status, text, error_type, error_message = "error", "", "", ""
                try:
                    text = clean_extracted_text(
                        page.extract_text() or "", config.get("text_cleaning", {})
                    )
                    status = classify_page_text(text, page_limit)
                except Exception as exc:
                    error_type, error_message = exc.__class__.__name__, str(exc)
                    errors.append(_error(meta, "page_extraction", exc, number))
                common = {
                    "source_id": meta["source_id"], "title": meta["title"],
                    "screening_decision": meta["screening_decision"],
                    "local_path": meta["local_path"],
                    "source_checksum_sha256": checksum, "page_number": number,
                    "extraction_status": status, "character_count": len(text),
                    "word_count": _words(text),
                }
                page_row = {**common, "text": text}
                pages.append(page_row)
                local_pages.append(page_row)
                diagnostics.append({
                    "source_id": meta["source_id"], "page_number": number,
                    "extraction_status": status, "character_count": len(text),
                    "word_count": _words(text),
                    "possible_scanned_page": _bool(status in {"empty", "low_text"}),
                    "error_type": error_type, "error_message": error_message,
                })

            status_counts = pd.Series(
                [page["extraction_status"] for page in local_pages], dtype="object"
            ).value_counts()
            with_text = int(status_counts.get("text", 0))
            low_text = int(status_counts.get("low_text", 0))
            empty = int(status_counts.get("empty", 0))
            failed = int(status_counts.get("error", 0))
            total_chars = sum(page["character_count"] for page in local_pages)
            total_words = sum(page["word_count"] for page in local_pages)
            denominator = inventory["page_count"] or 1
            weak_fraction = (low_text + empty + failed) / denominator
            needs_ocr = inventory["page_count"] > 0 and (
                weak_fraction >= scan_limit or total_chars < document_limit
            )
            inventory.update(
                pages_with_text=with_text, pages_low_text=low_text,
                pages_empty=empty, pages_error=failed,
                total_characters=total_chars, total_words=total_words,
                text_page_fraction=round(with_text / denominator, 6),
                low_or_empty_fraction=round(weak_fraction, 6),
                ocr_recommended=_bool(needs_ocr),
                parsing_status="parsed_with_page_errors" if failed else "parsed",
            )
        except Exception as exc:
            inventory.update(parsing_status="failed", notes=str(exc))
            errors.append(_error(meta, "document_parsing", exc))
        inventories.append(inventory)

    return (
        pd.DataFrame(inventories, columns=INVENTORY_COLUMNS),
        pd.DataFrame(pages, columns=PAGE_TEXT_COLUMNS),
        pd.DataFrame(diagnostics, columns=PAGE_DIAGNOSTIC_COLUMNS),
        pd.DataFrame(errors, columns=ERROR_COLUMNS),
    )


def _blocks(text: str) -> list[str]:
    items = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    if len(items) == 1 and "\n" in text:
        items = [item.strip() for item in text.splitlines() if item.strip()]
    return items


def segment_document_sections(
    page_text: pd.DataFrame, config: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect headings and assign text blocks to page-traceable sections."""
    if page_text.empty:
        return pd.DataFrame(columns=SECTION_COLUMNS), pd.DataFrame(columns=BLOCK_COLUMNS)
    sections, output_blocks = [], []
    settings = config["section_detection"]

    for source_id, source_pages in page_text.groupby("source_id", sort=False):
        source_pages = source_pages.sort_values("page_number")
        first = source_pages.iloc[0]
        section_index, block_index = 0, 0
        heading, method = "Unsectioned", "default"
        section_id = _stable_id("sec", source_id, section_index, heading)
        current = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            text = "\n\n".join(item["text"] for item in current if item["text"])
            if text:
                page_numbers = [int(item["page_number"]) for item in current]
                sections.append({
                    "section_id": section_id, "source_id": source_id,
                    "title": _clean(first.get("title")),
                    "screening_decision": _clean(first.get("screening_decision")),
                    "local_path": _clean(first.get("local_path")),
                    "source_checksum_sha256": _clean(first.get("source_checksum_sha256")),
                    "section_index": section_index, "section_heading": heading,
                    "heading_detection_method": method,
                    "start_page": min(page_numbers), "end_page": max(page_numbers),
                    "character_count": len(text), "word_count": _words(text),
                    "text": text,
                })
            current = []

        for _, page in source_pages.iterrows():
            if _clean(page.get("extraction_status")) == "error":
                continue
            for block in _blocks(_clean(page.get("text"))):
                is_heading, detected_by = detect_heading(block, settings)
                if is_heading:
                    flush()
                    section_index += 1
                    heading, method = block, detected_by
                    section_id = _stable_id("sec", source_id, section_index, heading)
                    continue
                block_index += 1
                row = {
                    "source_id": source_id, "title": _clean(first.get("title")),
                    "screening_decision": _clean(first.get("screening_decision")),
                    "local_path": _clean(first.get("local_path")),
                    "source_checksum_sha256": _clean(first.get("source_checksum_sha256")),
                    "section_id": section_id, "section_index": section_index,
                    "section_heading": heading, "heading_detection_method": method,
                    "page_number": int(page["page_number"]),
                    "block_index": block_index, "text": block,
                }
                current.append(row)
                output_blocks.append(row.copy())
        flush()

    return (
        pd.DataFrame(sections, columns=SECTION_COLUMNS),
        pd.DataFrame(output_blocks, columns=BLOCK_COLUMNS),
    )


def _section_text(blocks: pd.DataFrame) -> tuple[str, list[tuple[int, int, int]]]:
    parts, spans, cursor = [], [], 0
    for _, block in blocks.sort_values("block_index").iterrows():
        value = _clean(block.get("text"))
        if not value:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(value)
        cursor += len(value)
        spans.append((start, cursor, int(block["page_number"])))
    return "".join(parts), spans


def _chunk_end(text: str, start: int, target: int, fraction: float) -> int:
    desired = min(len(text), start + target)
    if desired == len(text):
        return desired
    lower = start + max(1, int(target * fraction))
    candidates = [text.rfind("\n\n", lower, desired)]
    candidates += [text.rfind(mark, lower, desired) for mark in (". ", "? ", "! ")]
    candidates += [text.rfind(" ", lower, desired)]
    boundary = max(candidates)
    return boundary + 1 if boundary > start else desired


def chunk_section_blocks(
    section_blocks: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    """Create overlapping chunks with section and page provenance."""
    if section_blocks.empty:
        return pd.DataFrame(columns=CHUNK_COLUMNS)
    settings = config["chunking"]
    target = int(settings["target_characters"])
    overlap = int(settings["overlap_characters"])
    minimum = int(settings["minimum_chunk_characters"])
    fraction = float(settings.get("minimum_break_fraction", 0.60))
    rows = []

    for section_id, blocks in section_blocks.groupby("section_id", sort=False):
        blocks = blocks.sort_values("block_index")
        first = blocks.iloc[0]
        text, spans = _section_text(blocks)
        raw, start = [], 0
        while start < len(text):
            end = _chunk_end(text, start, target, fraction)
            actual_start, actual_end = start, end
            while actual_start < actual_end and text[actual_start].isspace():
                actual_start += 1
            while actual_end > actual_start and text[actual_end - 1].isspace():
                actual_end -= 1
            value = text[actual_start:actual_end]
            if value:
                raw.append((actual_start, actual_end, value))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
            while start < len(text) and text[start].isspace():
                start += 1
        if len(raw) > 1 and len(raw[-1][2]) < minimum:
            previous, final = raw[-2], raw[-1]
            raw[-2] = (previous[0], final[1], text[previous[0]:final[1]].strip())
            raw.pop()

        for index, (start, end, value) in enumerate(raw, start=1):
            pages = [page for a, b, page in spans if b > start and a < end]
            rows.append({
                "chunk_id": _stable_id(
                    "chk", first["source_id"], section_id, index, value
                ),
                "source_id": _clean(first.get("source_id")),
                "title": _clean(first.get("title")),
                "screening_decision": _clean(first.get("screening_decision")),
                "local_path": _clean(first.get("local_path")),
                "source_checksum_sha256": _clean(first.get("source_checksum_sha256")),
                "section_id": section_id, "section_index": int(first["section_index"]),
                "section_heading": _clean(first.get("section_heading")),
                "chunk_index": index, "start_page": min(pages) if pages else 0,
                "end_page": max(pages) if pages else 0,
                "character_start": start, "character_end": end,
                "character_count": len(value), "word_count": _words(value),
                "text": value,
            })
    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)


def write_document_text_files(
    page_text: pd.DataFrame,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Write one page-marked UTF-8 text file per parsed source."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve() if project_root else None
    rows = []
    for source_id, pages in page_text.groupby("source_id", sort=False):
        path = directory / f"{source_id}.txt"
        content = "\n\n".join(
            f"[[PAGE {int(page['page_number'])} | {page['extraction_status']}]]\n"
            f"{_clean(page.get('text'))}"
            for _, page in pages.sort_values("page_number").iterrows()
        )
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        resolved = path.resolve()
        try:
            relative = str(resolved.relative_to(root)) if root else str(resolved)
        except ValueError:
            relative = str(resolved)
        rows.append({"source_id": source_id, "text_path": relative})
    return pd.DataFrame(rows, columns=["source_id", "text_path"])


def build_ocr_queue(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(columns=OCR_COLUMNS)
    queue = inventory.loc[inventory["ocr_recommended"].eq("true")].copy()
    return queue.reindex(columns=OCR_COLUMNS).reset_index(drop=True)


def build_full_text_screening_corpus(
    inventory: pd.DataFrame, page_text: pd.DataFrame
) -> pd.DataFrame:
    """Build one page-marked record per document for full-text screening."""
    groups = {
        source_id: pages.sort_values("page_number")
        for source_id, pages in page_text.groupby("source_id", sort=False)
    }
    rows = []
    for _, document in inventory.iterrows():
        source_id = _clean(document.get("source_id"))
        pages = groups.get(source_id, pd.DataFrame())
        full_text = "\n\n".join(
            f"[[PAGE {int(page['page_number'])}]]\n{_clean(page.get('text'))}"
            for _, page in pages.iterrows()
            if _clean(page.get("text"))
        )
        decision = _clean(document.get("screening_decision"))
        rows.append({
            "source_id": source_id, "title": _clean(document.get("title")),
            "screening_decision": decision,
            "full_text_screening_required": _bool(decision == "uncertain"),
            "local_path": _clean(document.get("local_path")),
            "source_checksum_sha256": _clean(document.get("observed_checksum_sha256")),
            "page_count": document.get("page_count", 0),
            "total_characters": document.get("total_characters", 0),
            "ocr_recommended": _clean(document.get("ocr_recommended")),
            "parsing_status": _clean(document.get("parsing_status")),
            "full_text": full_text,
        })
    return pd.DataFrame(rows, columns=SCREENING_CORPUS_COLUMNS)


def validate_parsing_outputs(
    inventory: pd.DataFrame,
    page_text: pd.DataFrame,
    sections: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Validate cross-table identities, counts, and page ranges."""
    issues = []

    def add(source_id, output, row_number, field, issue):
        issues.append({
            "source_id": _clean(source_id), "output": output,
            "row_number": row_number, "field": field, "issue": issue,
        })

    if "source_id" in inventory:
        duplicates = inventory["source_id"].astype(str).duplicated(keep=False)
        for index, row in inventory.loc[duplicates].iterrows():
            add(row["source_id"], "inventory", index + 2, "source_id", "duplicate")
    for index, document in inventory.fillna("").iterrows():
        source_id = document.get("source_id", "")
        source_pages = (
            page_text.loc[page_text["source_id"].eq(source_id)]
            if "source_id" in page_text.columns else pd.DataFrame()
        )
        expected = int(document.get("page_count") or 0)
        if (
            len(source_pages) != expected
            and _clean(document.get("parsing_status")).startswith("parsed")
        ):
            add(
                source_id, "inventory", index + 2, "page_count",
                f"inventory={expected};page_text={len(source_pages)}",
            )
        if _clean(document.get("checksum_matches")) == "false":
            add(
                source_id, "inventory", index + 2, "checksum_matches",
                "checksum_mismatch",
            )
    if not page_text.empty:
        duplicates = page_text[["source_id", "page_number"]].astype(str).duplicated(
            keep=False
        )
        for index in page_text.index[duplicates]:
            row = page_text.loc[index]
            add(
                row["source_id"], "page_text", index + 2, "page_number",
                "duplicate_source_page",
            )
    for output, frame in (("sections", sections), ("chunks", chunks)):
        if frame.empty:
            continue
        if output == "chunks":
            duplicates = frame["chunk_id"].astype(str).duplicated(keep=False)
            for index, row in frame.loc[duplicates].iterrows():
                add(row["source_id"], output, index + 2, "chunk_id", "duplicate")
        for index, row in frame.fillna("").iterrows():
            if not _clean(row.get("text")):
                add(row.get("source_id"), output, index + 2, "text", "empty_text")
            if int(row.get("start_page") or 0) > int(row.get("end_page") or 0):
                add(
                    row.get("source_id"), output, index + 2, "page_range",
                    "start_page_after_end_page",
                )
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def parsing_summary(
    inventory: pd.DataFrame,
    page_text: pd.DataFrame,
    sections: pd.DataFrame,
    chunks: pd.DataFrame,
    errors: pd.DataFrame,
) -> pd.DataFrame:
    status = inventory.get(
        "parsing_status", pd.Series("", index=inventory.index, dtype="object")
    ).astype(str)
    ocr = inventory.get(
        "ocr_recommended", pd.Series("", index=inventory.index, dtype="object")
    ).astype(str)
    return pd.DataFrame({
        "metric": [
            "documents_in_manifest", "documents_parsed", "documents_failed",
            "documents_requiring_ocr", "pages_extracted", "sections_detected",
            "chunks_created", "errors_recorded",
        ],
        "value": [
            len(inventory), int(status.str.startswith("parsed").sum()),
            int(status.eq("failed").sum()), int(ocr.eq("true").sum()),
            len(page_text), len(sections), len(chunks), len(errors),
        ],
    })


def run_document_parsing(
    manifest: pd.DataFrame,
    config: Mapping[str, object],
    *,
    project_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Run the local parsing pipeline without OCR or network calls."""
    root = Path(project_root).resolve()
    inventory, page_text, diagnostics, errors = extract_manifest_documents(
        manifest, config, project_root=root
    )
    text_paths = write_document_text_files(
        page_text, root / str(config["paths"]["text_dir"]), project_root=root
    )
    if not text_paths.empty:
        inventory["text_path"] = inventory["source_id"].map(
            text_paths.set_index("source_id")["text_path"]
        ).fillna("")
    sections, blocks = segment_document_sections(page_text, config)
    chunks = chunk_section_blocks(blocks, config)
    return {
        "inventory": inventory,
        "page_text": page_text,
        "page_diagnostics": diagnostics,
        "sections": sections,
        "chunks": chunks,
        "errors": errors,
        "issues": validate_parsing_outputs(inventory, page_text, sections, chunks),
        "ocr_queue": build_ocr_queue(inventory),
        "screening_corpus": build_full_text_screening_corpus(inventory, page_text),
    }
