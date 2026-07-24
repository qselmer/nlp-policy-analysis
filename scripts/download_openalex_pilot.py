"""Descarga un corpus piloto desde OpenAlex y registra la búsqueda.

El script utiliza la consulta inglesa prioritaria generada por
``notebooks/02_search_strategy.ipynb``. La consulta conceptual se adapta a la
sintaxis de OpenAlex antes de ejecutarse.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

PLANNED_QUERIES_PATH = ROOT / "data" / "interim" / "planned_queries.csv"
SEARCH_LOG_PATH = ROOT / "data" / "interim" / "search_log_working.csv"
IMPORT_MANIFEST_PATH = ROOT / "data" / "interim" / "import_manifest_working.csv"

EXPORT_DIR = ROOT / "data" / "raw" / "search_exports"
EXPORT_PATH = EXPORT_DIR / "openalex_pilot_en.csv"

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
PILOT_SIZE = 50
OPENALEX_FILTER = "from_publication_date:1990-01-01,type:article"

# OpenAlex no admite comodines dentro de algunas frases exactas.
OPENALEX_QUERY_REPLACEMENTS = {
    '"marine heatwave*"': '("marine heatwave" OR "marine heatwaves")',
    '"regime shift*"': '("regime shift" OR "regime shifts")',
    '"small pelagic*"': '("small pelagic" OR "small pelagics")',
}


def adapt_query_for_openalex(query: str) -> str:
    """Convierte la consulta conceptual a una consulta válida en OpenAlex."""
    adapted = str(query)
    for original, replacement in OPENALEX_QUERY_REPLACEMENTS.items():
        adapted = adapted.replace(original, replacement)
    return adapted


def reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    """Reconstruye un resumen desde el índice invertido de OpenAlex."""
    if not inverted_index:
        return ""

    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((int(position), str(word)))

    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def extract_authors(authorships: list[dict[str, Any]] | None) -> str:
    """Convierte la estructura de autorías de OpenAlex en texto legible."""
    names: list[str] = []
    for authorship in authorships or []:
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            names.append(name)
    return "; ".join(names)


def load_priority_query() -> str:
    """Lee la consulta inglesa ``priority_taxa`` generada en la fase 02."""
    if not PLANNED_QUERIES_PATH.exists():
        raise FileNotFoundError(
            "No existe data/interim/planned_queries.csv. "
            "Ejecuta primero notebooks/02_search_strategy.ipynb."
        )

    planned = pd.read_csv(PLANNED_QUERIES_PATH)
    required = {"language", "query_type", "exact_query"}
    missing = required.difference(planned.columns)
    if missing:
        raise ValueError(
            "planned_queries.csv no contiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    selected = planned.loc[
        (planned["language"] == "en")
        & (planned["query_type"] == "priority_taxa")
    ]

    if len(selected) != 1:
        raise ValueError(
            "Debe existir exactamente una consulta en_priority_taxa."
        )

    query = str(selected.iloc[0]["exact_query"]).strip()
    if not query:
        raise ValueError("La consulta en_priority_taxa está vacía.")
    return query


def _safe_openalex_error(response: requests.Response) -> str:
    """Extrae un mensaje de error sin incluir la URL ni la clave de API."""
    error_message = "Respuesta inválida sin detalle adicional."

    try:
        payload = response.json()
    except ValueError:
        text = str(response.text or "").strip()
        if text:
            error_message = text
    else:
        if isinstance(payload, dict):
            detail = (
                payload.get("message")
                or payload.get("error")
                or payload.get("detail")
                or payload
            )
        else:
            detail = payload

        if isinstance(detail, str):
            error_message = detail
        else:
            error_message = json.dumps(
                detail,
                ensure_ascii=False,
                default=str,
            )

    return error_message[:1500]


def download_openalex(
    query: str,
    api_key: str,
) -> tuple[pd.DataFrame, int]:
    """Ejecuta la búsqueda piloto y devuelve registros tabulares."""
    params = {
        "api_key": api_key,
        # Los comodines fisher* y anchov* requieren búsqueda exacta.
        "search.exact": query,
        "filter": OPENALEX_FILTER,
        "per_page": PILOT_SIZE,
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "display_name",
                "publication_year",
                "publication_date",
                "type",
                "language",
                "authorships",
                "primary_location",
                "open_access",
                "abstract_inverted_index",
                "cited_by_count",
                "is_retracted",
            ]
        ),
    }

    try:
        response = requests.get(
            OPENALEX_ENDPOINT,
            params=params,
            timeout=90,
        )
    except requests.RequestException as exc:
        # No se muestra ``exc`` porque puede incluir la URL con la clave API.
        raise RuntimeError(
            "No se pudo conectar con OpenAlex. Revisa la conexión e inténtalo "
            "nuevamente."
        ) from None

    if not response.ok:
        error_message = _safe_openalex_error(response)
        raise RuntimeError(
            f"OpenAlex devolvió HTTP {response.status_code}: {error_message}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "OpenAlex devolvió una respuesta que no es JSON válido."
        ) from exc

    works = payload.get("results") or []
    total_results = int((payload.get("meta") or {}).get("count") or 0)

    if not works:
        raise RuntimeError(
            "La consulta no devolvió registros. Revisa la clave, los filtros "
            "y la sintaxis de búsqueda."
        )

    rows: list[dict[str, Any]] = []
    for work in works:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        open_access = work.get("open_access") or {}
        landing_page = str(
            location.get("landing_page_url")
            or work.get("id")
            or ""
        )

        rows.append(
            {
                "title": work.get("title") or work.get("display_name") or "",
                "authors": extract_authors(work.get("authorships")),
                "publisher": source.get("display_name") or "",
                "publication_year": work.get("publication_year") or "",
                "publication_date": work.get("publication_date") or "",
                "doi": work.get("doi") or "",
                "primary_url": landing_page,
                "landing_page_url": landing_page,
                "abstract": reconstruct_abstract(
                    work.get("abstract_inverted_index")
                ),
                "language": work.get("language") or "",
                "openalex_id": work.get("id") or "",
                "openalex_type": work.get("type") or "article",
                "oa_status": open_access.get("oa_status") or "",
                "cited_by_count": work.get("cited_by_count") or 0,
                "is_retracted": bool(work.get("is_retracted", False)),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.loc[~frame["is_retracted"]].reset_index(drop=True)
    return frame, total_results


def update_search_log(
    *,
    exact_query: str,
    total_results: int,
    exported_records: int,
) -> None:
    """Registra la búsqueda efectivamente ejecutada."""
    if not SEARCH_LOG_PATH.exists():
        raise FileNotFoundError(
            "No existe search_log_working.csv. "
            "Ejecuta notebooks/02_search_strategy.ipynb."
        )

    log = pd.read_csv(SEARCH_LOG_PATH)
    today = date.today().isoformat()
    compact_date = date.today().strftime("%Y%m%d")
    search_id = f"{compact_date}_openalex_en_priority_taxa_pilot"

    row = {
        "search_id": search_id,
        "search_date": today,
        "reviewer": "Elmer Quispe-Salazar",
        "source_channel": "scientific_database",
        "platform_or_website": "OpenAlex",
        "database_collection": "Works",
        "language": "en",
        "query_type": "priority_taxa",
        "exact_query": exact_query,
        "filters_applied": OPENALEX_FILTER,
        "date_from": 1990,
        "date_to": "",
        "result_count": total_results,
        "export_filename": EXPORT_PATH.name,
        "export_format": "csv",
        "search_status": "completed",
        "notes": (
            f"Pilot export. Retained {exported_records} records from the "
            f"first {PILOT_SIZE} OpenAlex results."
        ),
    }

    if "search_id" in log.columns:
        log = log.loc[log["search_id"] != search_id]

    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(SEARCH_LOG_PATH, index=False, encoding="utf-8-sig")


def update_import_manifest() -> None:
    """Registra el archivo para su procesamiento en la fase 03."""
    if not IMPORT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No existe import_manifest_working.csv. "
            "Ejecuta notebooks/03_import_and_deduplicate_sources.ipynb."
        )

    manifest = pd.read_csv(IMPORT_MANIFEST_PATH)
    row = {
        "file_name": EXPORT_PATH.name,
        "platform": "openalex",
        "source_type": "scientific_article",
        "source_status": "other",
        "source_language": "en",
        "notes": (
            "OpenAlex English priority-taxa pilot; peer-review status pending "
            "verification."
        ),
    }

    if "file_name" in manifest.columns:
        manifest = manifest.loc[
            manifest["file_name"] != EXPORT_PATH.name
        ]

    manifest = pd.concat(
        [manifest, pd.DataFrame([row])],
        ignore_index=True,
    )
    manifest.to_csv(
        IMPORT_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    """Ejecuta la descarga y actualiza los registros de auditoría."""
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "No se encontró OPENALEX_API_KEY en la sesión de PowerShell."
        )

    conceptual_query = load_priority_query()
    exact_query = adapt_query_for_openalex(conceptual_query)

    print("Consulta conceptual:")
    print(conceptual_query)
    print()
    print("Consulta adaptada para OpenAlex:")
    print(exact_query)
    print()

    frame, total_results = download_openalex(
        query=exact_query,
        api_key=api_key,
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXPORT_PATH, index=False, encoding="utf-8-sig")

    update_search_log(
        exact_query=exact_query,
        total_results=total_results,
        exported_records=len(frame),
    )
    update_import_manifest()

    print(f"Total encontrado por OpenAlex: {total_results}")
    print(f"Registros exportados: {len(frame)}")
    print(f"Archivo: {EXPORT_PATH.relative_to(ROOT)}")
    print(f"Search log: {SEARCH_LOG_PATH.relative_to(ROOT)}")
    print(f"Import manifest: {IMPORT_MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
