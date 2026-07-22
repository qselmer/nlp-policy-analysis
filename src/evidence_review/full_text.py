"""Full-text retrieval and registration for the climate-fisheries evidence map."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import mimetypes
from pathlib import Path
import re
from typing import Iterable, Mapping
from urllib.parse import quote, urlparse

import pandas as pd
import requests
import yaml

from evidence_review.importing import normalise_doi


RETRIEVAL_COLUMNS = [
    "source_id",
    "title",
    "abstract",
    "authors_or_organisation",
    "publisher",
    "year",
    "source_type",
    "source_status",
    "source_language",
    "doi",
    "primary_url",
    "landing_page_url",
    "screening_decision",
    "full_text_screening_required",
    "openalex_id",
    "openalex_lookup_status",
    "openalex_lookup_date",
    "is_open_access",
    "oa_status",
    "best_landing_page_url",
    "best_pdf_url",
    "license",
    "version",
    "host_source",
    "retrieval_status",
    "retrieval_method",
    "retrieval_url",
    "access_date",
    "local_path",
    "file_name",
    "mime_type",
    "file_size_bytes",
    "checksum_sha256",
    "pdf_signature_valid",
    "manual_action",
    "retrieval_notes",
]

PRESERVED_COLUMNS = RETRIEVAL_COLUMNS[14:]

MANIFEST_COLUMNS = [
    "source_id",
    "title",
    "screening_decision",
    "retrieval_status",
    "retrieval_method",
    "retrieval_url",
    "access_date",
    "local_path",
    "file_name",
    "mime_type",
    "file_size_bytes",
    "checksum_sha256",
    "pdf_signature_valid",
    "license",
    "version",
]

ERROR_COLUMNS = [
    "source_id",
    "stage",
    "error_type",
    "error_message",
]


def load_full_text_config(
    path: str | Path = "config/full_text_retrieval.yml",
) -> dict:
    """Load and minimally validate full-text retrieval configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("full-text retrieval configuration must be a YAML mapping")
    required = {"retrieval_statuses", "paths", "openalex", "download"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing full-text configuration sections: {missing}")
    return config


def read_csv_robust(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV using common UTF-8 and Windows encodings."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Unable to decode CSV: " + " | ".join(errors))


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _as_bool_text(value: object) -> str:
    text = _clean(value).casefold()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return ""


def build_retrieval_queue(
    included: pd.DataFrame,
    uncertain: pd.DataFrame,
) -> pd.DataFrame:
    """Combine included and uncertain records into a unique full-text queue."""
    frames: list[pd.DataFrame] = []
    for decision, frame in (("include", included), ("uncertain", uncertain)):
        if frame is None or frame.empty:
            continue
        part = frame.copy()
        part["screening_decision"] = decision
        part["full_text_screening_required"] = decision == "uncertain"
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)

    queue = pd.concat(frames, ignore_index=True)
    if "source_id" not in queue:
        raise ValueError("screening exports must include source_id")
    queue["source_id"] = queue["source_id"].map(_clean)
    if queue["source_id"].eq("").any():
        raise ValueError("screening exports contain blank source_id values")
    queue = queue.drop_duplicates("source_id", keep="first").reset_index(drop=True)

    metadata_columns = RETRIEVAL_COLUMNS[:14]
    for column in metadata_columns:
        if column not in queue:
            queue[column] = ""
    queue["doi"] = queue["doi"].map(normalise_doi)
    queue["full_text_screening_required"] = queue[
        "full_text_screening_required"
    ].map(lambda value: "true" if bool(value) else "false")
    for column in PRESERVED_COLUMNS:
        queue[column] = ""
    return queue[RETRIEVAL_COLUMNS].copy()


def initialise_retrieval_sheet(
    queue: pd.DataFrame,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refresh metadata while preserving prior retrieval work by source_id."""
    sheet = queue.copy()
    for column in RETRIEVAL_COLUMNS:
        if column not in sheet:
            sheet[column] = ""
    if existing is None or existing.empty or "source_id" not in existing:
        return sheet[RETRIEVAL_COLUMNS].copy()

    prior = existing.drop_duplicates("source_id", keep="last").set_index("source_id")
    for column in PRESERVED_COLUMNS:
        if column not in prior:
            continue
        values = sheet["source_id"].map(prior[column])
        mask = values.notna()
        sheet.loc[mask, column] = values.loc[mask].astype(str)
    return sheet[RETRIEVAL_COLUMNS].copy()


def _location_payload(location: object) -> dict:
    return location if isinstance(location, dict) else {}


def parse_openalex_work(work: Mapping[str, object]) -> dict[str, str]:
    """Flatten retrieval-relevant fields from one OpenAlex work object."""
    best = _location_payload(work.get("best_oa_location"))
    primary = _location_payload(work.get("primary_location"))
    open_access = work.get("open_access")
    open_access = open_access if isinstance(open_access, dict) else {}

    chosen = best or primary
    source = chosen.get("source")
    source = source if isinstance(source, dict) else {}

    is_oa = chosen.get("is_oa")
    if is_oa is None:
        is_oa = open_access.get("is_oa")

    return {
        "openalex_id": _clean(work.get("id")),
        "is_open_access": _as_bool_text(is_oa),
        "oa_status": _clean(open_access.get("oa_status")),
        "best_landing_page_url": _clean(chosen.get("landing_page_url")),
        "best_pdf_url": _clean(chosen.get("pdf_url")),
        "license": _clean(chosen.get("license") or chosen.get("license_id")),
        "version": _clean(chosen.get("version")),
        "host_source": _clean(source.get("display_name")),
    }


@dataclass(frozen=True)
class OpenAlexLookupResult:
    fields: dict[str, str]
    status: str
    error: str = ""


def lookup_openalex_by_doi(
    doi: object,
    *,
    api_key: str,
    base_url: str = "https://api.openalex.org",
    timeout_seconds: int = 30,
    session: requests.Session | None = None,
) -> OpenAlexLookupResult:
    """Look up one work by DOI without exposing the API key in errors."""
    doi_value = normalise_doi(doi)
    if not doi_value:
        return OpenAlexLookupResult(fields={}, status="missing_doi")
    if not api_key:
        return OpenAlexLookupResult(fields={}, status="missing_api_key")

    identifier = quote(f"doi:{doi_value}", safe="")
    url = f"{base_url.rstrip('/')}/works/{identifier}"
    client = session or requests.Session()

    try:
        response = client.get(
            url,
            params={"api_key": api_key},
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )
    except requests.RequestException as exc:
        return OpenAlexLookupResult(
            fields={},
            status="request_error",
            error=exc.__class__.__name__,
        )

    if response.status_code == 404:
        return OpenAlexLookupResult(fields={}, status="not_found")
    if response.status_code == 429:
        return OpenAlexLookupResult(fields={}, status="rate_limited")
    if response.status_code != 200:
        return OpenAlexLookupResult(
            fields={},
            status=f"http_{response.status_code}",
        )

    try:
        payload = response.json()
    except ValueError:
        return OpenAlexLookupResult(fields={}, status="invalid_json")

    if not isinstance(payload, dict):
        return OpenAlexLookupResult(fields={}, status="invalid_payload")
    return OpenAlexLookupResult(
        fields=parse_openalex_work(payload),
        status="resolved",
    )


def classify_retrieval_status(row: Mapping[str, object]) -> str:
    """Classify availability without treating a landing page as full text."""
    local_path = _clean(row.get("local_path"))
    pdf_url = _clean(row.get("best_pdf_url"))
    landing = _clean(
        row.get("best_landing_page_url")
        or row.get("landing_page_url")
        or row.get("primary_url")
    )
    is_oa = _as_bool_text(row.get("is_open_access"))
    lookup_status = _clean(row.get("openalex_lookup_status"))

    if local_path:
        return "available_local"
    if pdf_url and is_oa == "true":
        return "available_open_access"
    if pdf_url and is_oa != "true":
        return "restricted_access"
    if landing:
        return "landing_page_only"
    if lookup_status in {
        "request_error",
        "rate_limited",
        "invalid_json",
        "invalid_payload",
    }:
        return "retrieval_error"
    return "not_found"


def resolve_openalex_queue(
    sheet: pd.DataFrame,
    *,
    api_key: str,
    config: Mapping[str, object],
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve DOI metadata for a queue and return an error audit."""
    output = sheet.copy()
    errors: list[dict[str, str]] = []
    openalex = config["openalex"]
    lookup_date = date.today().isoformat()

    for index, row in output.iterrows():
        if not overwrite and _clean(row.get("openalex_lookup_status")) == "resolved":
            continue
        result = lookup_openalex_by_doi(
            row.get("doi"),
            api_key=api_key,
            base_url=str(openalex["base_url"]),
            timeout_seconds=int(openalex["timeout_seconds"]),
            session=session,
        )
        output.at[index, "openalex_lookup_status"] = result.status
        output.at[index, "openalex_lookup_date"] = lookup_date
        for field, value in result.fields.items():
            output.at[index, field] = value
        output.at[index, "retrieval_status"] = classify_retrieval_status(
            output.loc[index].to_dict()
        )
        if result.error or result.status not in {
            "resolved",
            "missing_doi",
            "not_found",
        }:
            errors.append(
                {
                    "source_id": _clean(row.get("source_id")),
                    "stage": "openalex_lookup",
                    "error_type": result.status,
                    "error_message": result.error,
                }
            )

    return output, pd.DataFrame(errors, columns=ERROR_COLUMNS)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading the entire file into memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_pdf_signature(path: str | Path) -> bool:
    with Path(path).open("rb") as stream:
        return stream.read(5) == b"%PDF-"


def safe_pdf_filename(source_id: object, title: object = "") -> str:
    """Create a deterministic, filesystem-safe PDF filename."""
    source = re.sub(r"[^A-Za-z0-9_-]+", "_", _clean(source_id)).strip("_")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", _clean(title)).strip("-").lower()
    slug = slug[:80].rstrip("-")
    return f"{source}__{slug or 'full-text'}.pdf"


def register_local_files(
    sheet: pd.DataFrame,
    full_text_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Register files whose names begin with their source_id."""
    output = sheet.copy()
    directory = Path(full_text_dir)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve() if project_root else None
    errors: list[dict[str, str]] = []

    all_files = [path for path in directory.rglob("*") if path.is_file()]
    by_prefix: dict[str, list[Path]] = {}
    for path in all_files:
        prefix = path.name.split("__", 1)[0].split(".", 1)[0]
        by_prefix.setdefault(prefix, []).append(path)

    for index, row in output.iterrows():
        source_id = _clean(row.get("source_id"))
        candidates = sorted(by_prefix.get(source_id, []))
        if not candidates:
            if not _clean(row.get("retrieval_status")):
                output.at[index, "retrieval_status"] = classify_retrieval_status(
                    output.loc[index].to_dict()
                )
            continue
        path = candidates[0]
        if len(candidates) > 1:
            errors.append(
                {
                    "source_id": source_id,
                    "stage": "local_registration",
                    "error_type": "multiple_local_files",
                    "error_message": " | ".join(str(item) for item in candidates),
                }
            )
        resolved = path.resolve()
        try:
            local_path = str(resolved.relative_to(root)) if root else str(resolved)
        except ValueError:
            local_path = str(resolved)

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        output.at[index, "retrieval_status"] = "available_local"
        output.at[index, "retrieval_method"] = (
            _clean(row.get("retrieval_method")) or "local_file"
        )
        output.at[index, "local_path"] = local_path
        output.at[index, "file_name"] = path.name
        output.at[index, "mime_type"] = mime_type
        output.at[index, "file_size_bytes"] = str(path.stat().st_size)
        output.at[index, "checksum_sha256"] = file_sha256(path)
        output.at[index, "pdf_signature_valid"] = (
            "true"
            if path.suffix.lower() == ".pdf" and has_pdf_signature(path)
            else "false"
        )

    return output, pd.DataFrame(errors, columns=ERROR_COLUMNS)


def _validate_download_url(url: str, allowed_schemes: Iterable[str]) -> None:
    parsed = urlparse(url)
    allowed = {item.casefold() for item in allowed_schemes}
    if parsed.scheme.casefold() not in allowed:
        raise ValueError("download URL uses a disallowed scheme")
    if not parsed.netloc:
        raise ValueError("download URL has no host")


def download_open_access_pdf(
    row: Mapping[str, object],
    target_dir: str | Path,
    *,
    config: Mapping[str, object],
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Download one explicitly open-access PDF with safety checks."""
    if _clean(row.get("retrieval_status")) != "available_open_access":
        raise ValueError("only available_open_access records may be downloaded")
    if _as_bool_text(row.get("is_open_access")) != "true":
        raise ValueError("open-access confirmation is required")

    url = _clean(row.get("best_pdf_url"))
    if not url:
        raise ValueError("best_pdf_url is required")
    download = config["download"]
    _validate_download_url(url, download["allowed_schemes"])

    directory = Path(target_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = safe_pdf_filename(row.get("source_id"), row.get("title"))
    final_path = directory / filename
    if final_path.exists() and not overwrite:
        raise FileExistsError(str(final_path))

    max_bytes = int(download["max_file_size_mb"]) * 1024 * 1024
    client = session or requests.Session()
    response = client.get(
        url,
        timeout=int(download["timeout_seconds"]),
        headers={
            "Accept": "application/pdf",
            "User-Agent": str(download["user_agent"]),
        },
        stream=True,
        allow_redirects=True,
    )
    if response.status_code != 200:
        raise RuntimeError(f"download failed with HTTP {response.status_code}")

    content_type = _clean(response.headers.get("Content-Type"))
    content_type = content_type.split(";", 1)[0].lower()
    allowed_types = {
        str(item).lower() for item in download["accepted_content_types"]
    }
    if content_type and content_type not in allowed_types:
        raise ValueError(f"unexpected content type: {content_type}")

    temporary = final_path.with_suffix(".part")
    total = 0
    try:
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("download exceeds configured maximum size")
                stream.write(chunk)
        if not has_pdf_signature(temporary):
            raise ValueError("downloaded content does not have a PDF signature")
        temporary.replace(final_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "retrieval_status": "available_local",
        "retrieval_method": "openalex_open_access_pdf",
        "retrieval_url": url,
        "access_date": date.today().isoformat(),
        "local_path": str(final_path),
        "file_name": final_path.name,
        "mime_type": "application/pdf",
        "file_size_bytes": str(final_path.stat().st_size),
        "checksum_sha256": file_sha256(final_path),
        "pdf_signature_valid": "true",
    }


def download_open_access_queue(
    sheet: pd.DataFrame,
    target_dir: str | Path,
    *,
    config: Mapping[str, object],
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download eligible open-access PDFs and retain row-level failures."""
    output = sheet.copy()
    errors: list[dict[str, str]] = []
    for index, row in output.iterrows():
        if _clean(row.get("retrieval_status")) != "available_open_access":
            continue
        try:
            fields = download_open_access_pdf(
                row.to_dict(),
                target_dir,
                config=config,
                session=session,
                overwrite=overwrite,
            )
            for field, value in fields.items():
                output.at[index, field] = value
        except Exception as exc:
            errors.append(
                {
                    "source_id": _clean(row.get("source_id")),
                    "stage": "pdf_download",
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
    return output, pd.DataFrame(errors, columns=ERROR_COLUMNS)


def validate_retrieval_sheet(
    sheet: pd.DataFrame,
    config: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return row-level retrieval issues without altering the queue."""
    issues: list[dict[str, object]] = []
    allowed_statuses = set(config["retrieval_statuses"])
    root = Path(project_root).resolve() if project_root else None

    if "source_id" not in sheet:
        return pd.DataFrame(
            [
                {
                    "row_number": 1,
                    "source_id": "",
                    "field": "source_id",
                    "issue": "missing_column",
                }
            ]
        )

    duplicated = sheet["source_id"].astype(str).duplicated(keep=False)
    for index, source_id in sheet.loc[duplicated, "source_id"].items():
        issues.append(
            {
                "row_number": index + 2,
                "source_id": source_id,
                "field": "source_id",
                "issue": "duplicate_source_id",
            }
        )

    for index, row in sheet.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        status = _clean(row.get("retrieval_status"))
        local_path = _clean(row.get("local_path"))
        pdf_url = _clean(row.get("best_pdf_url"))
        if status and status not in allowed_statuses:
            issues.append(
                {
                    "row_number": index + 2,
                    "source_id": source_id,
                    "field": "retrieval_status",
                    "issue": f"unknown_retrieval_status:{status}",
                }
            )
        if status == "available_local" and not local_path:
            issues.append(
                {
                    "row_number": index + 2,
                    "source_id": source_id,
                    "field": "local_path",
                    "issue": "available_local_requires_local_path",
                }
            )
        if status == "available_open_access" and not pdf_url:
            issues.append(
                {
                    "row_number": index + 2,
                    "source_id": source_id,
                    "field": "best_pdf_url",
                    "issue": "available_open_access_requires_pdf_url",
                }
            )
        if local_path and root:
            path = Path(local_path)
            candidate = path if path.is_absolute() else root / path
            if not candidate.exists():
                issues.append(
                    {
                        "row_number": index + 2,
                        "source_id": source_id,
                        "field": "local_path",
                        "issue": "local_file_not_found",
                    }
                )

    return pd.DataFrame(
        issues,
        columns=["row_number", "source_id", "field", "issue"],
    )


def build_full_text_manifest(sheet: pd.DataFrame) -> pd.DataFrame:
    """Return one row for each registered local full-text file."""
    if sheet.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = sheet.loc[
        sheet["retrieval_status"].eq("available_local")
        & sheet["local_path"].astype(str).str.strip().ne("")
    ].copy()
    for column in MANIFEST_COLUMNS:
        if column not in manifest:
            manifest[column] = ""
    return manifest[MANIFEST_COLUMNS].reset_index(drop=True)


def retrieval_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    """Summarise queue size, decisions, availability, and local files."""
    status = sheet.get(
        "retrieval_status", pd.Series("", index=sheet.index)
    ).fillna("").astype(str).str.strip()
    decision = sheet.get(
        "screening_decision", pd.Series("", index=sheet.index)
    ).fillna("").astype(str).str.strip()

    rows = [
        ("queue_total", len(sheet)),
        ("screening_include", int(decision.eq("include").sum())),
        ("screening_uncertain", int(decision.eq("uncertain").sum())),
        ("status_pending", int(status.eq("").sum())),
    ]
    for value in sorted(set(status) - {""}):
        rows.append((f"status_{value}", int(status.eq(value).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"])


def split_retrieval_outputs(sheet: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create stable output groups for subsequent workflow stages."""
    available = sheet.loc[
        sheet["retrieval_status"].isin(
            ["available_local", "available_open_access"]
        )
    ].copy()
    unavailable = sheet.loc[
        sheet["retrieval_status"].isin(
            [
                "restricted_access",
                "landing_page_only",
                "not_found",
                "retrieval_error",
            ]
        )
    ].copy()
    screening_queue = sheet.loc[
        sheet["screening_decision"].eq("uncertain")
    ].copy()
    pending = sheet.loc[
        sheet["retrieval_status"].astype(str).str.strip().eq("")
    ].copy()
    return {
        "available": available.reset_index(drop=True),
        "unavailable": unavailable.reset_index(drop=True),
        "screening_queue": screening_queue.reset_index(drop=True),
        "pending": pending.reset_index(drop=True),
    }
