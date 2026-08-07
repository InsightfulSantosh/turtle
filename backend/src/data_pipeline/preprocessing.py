"""Identifier and fabric normalization for source workbook rows."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

IDENTIFIER_COMPONENT_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z0-9]{4})"
    r"(?P<style>[A-Z0-9]+)"
    r"(?P<colour>[0-9]{4})"
    r"(?P<descriptor>[A-Z][A-Z0-9/]*)?$"
)
IDENTIFIER_PATTERN = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]+-[0-9]{4}$")

CONTROLLED_FABRICS = (
    "Cotton – 100%",
    "Cotton – Stretch",
    "Cotton – Polyester Blend",
    "Cotton – Polyester Stretch",
    "Cotton – Linen Blend",
    "Cotton – Viscose Blend",
    "Cotton – Modal Blend",
    "Cotton – Tencel Blend",
    "Cotton – Lyocell Blend",
    "Cotton – Nylon Blend",
    "Cotton – Nylon Stretch",
    "Cotton – Other Blend",
    "Polyester – 100%",
    "Polyester – Stretch",
    "Polyester – Viscose Blend",
    "Polyester – Viscose Stretch",
    "Polyester – Viscose Linen Blend",
    "Polyester – Viscose Linen Stretch",
    "Polyester – Cotton Blend",
    "Polyester – Cotton Stretch",
    "Polyester – Bamboo Blend",
    "Polyester – Bamboo Stretch",
    "Polyester – Other Blend",
    "Viscose – 100%",
    "Viscose – Nylon Blend",
    "Viscose – Nylon Stretch",
    "Nylon – 100%",
    "Nylon – Stretch",
    "Modal – 100%",
    "Tencel – 100%",
    "Lyocell – Blend",
    "Multi-Fibre Blend",
    "Unspecified",
    "Review Required",
)
CONTROLLED_FABRIC_SET = frozenset(CONTROLLED_FABRICS)


class PreprocessingError(ValueError):
    """Raised when a source value cannot be safely normalized."""


@dataclass(frozen=True)
class CleaningReport:
    dataset: str
    identifier_column: str
    rows: int
    identifier_values_changed: int
    unique_identifiers_before: int
    unique_identifiers_after: int
    identifier_collisions_introduced: int
    review_required_fabric: int
    unspecified_fabric: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper()
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: object) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.upper() in {"", "-", "NA", "N/A", "NONE"}:
            return 0.0
        if cleaned.endswith("%"):
            return float(cleaned.rstrip("%")) / 100
        return float(cleaned)
    return float(value)


def parse_integer(value: object) -> int:
    return int(round(parse_number(value)))


def clean_product_identifier(value: object) -> str:
    """Return a validated PREFIX-STYLE-COLOUR product identifier."""

    compact = re.sub(r"[-\s]+", "", normalize_text(value))
    parts = IDENTIFIER_COMPONENT_PATTERN.fullmatch(compact)
    if not parts:
        raise PreprocessingError(f"Unable to parse product identifier: {value!r}")
    return (
        f"{parts.group('prefix')}-"
        f"{parts.group('style')}-"
        f"{parts.group('colour')}"
    )


def has_stretch_marker(text: str) -> bool:
    if "STRETCH" in text or text.startswith("PVS"):
        return True
    return bool(
        re.search(
            r"(^|[^A-Z])(SPANDEX|ELASTANE|LYCRA|SPDX|SPX|SPD|ELA|EA|LYC|SP)"
            r"([^A-Z]|$)",
            text,
        )
    )


def standardize_fabric(value: object) -> str:
    """Map source fabric text to the controlled fabric-family vocabulary."""

    text = normalize_text(value)
    compact = text.replace(" ", "")

    if not text or text in {"AS PER SAMPLE", "KNITTED", "4 WAY STRETCH"}:
        return "Unspecified"

    if (
        "EXCEL" in text
        or "EXC" in text
        or (compact.startswith("PVL") and not text.startswith("PV LINEN"))
        or text == "TR"
        or text.startswith(("TR ", "T/R", "T/V", "86%T ", "C/T"))
    ):
        return "Review Required"

    stretch = has_stretch_marker(text)
    has_cotton = any(
        token in text
        for token in ("COTTON", "COOTON", "CTN", "COT", "C/", "/C(")
    )
    has_polyester = any(
        token in text
        for token in ("POLY", "POLLY", "POLYSTER", "P/", "PVS", "PV", "PC", "C/P")
    )
    has_viscose = any(
        token in text for token in ("VISCOSE", " VIS", "VIS/", "VIS(", "RAYON")
    )
    has_linen = "LINEN" in text or "LIN" in text
    has_nylon = "NYLON" in text or text.startswith(("R/N", "C/N"))
    has_bamboo = "BAMBOO" in text
    has_modal = "MODAL" in text
    has_tencel = "TENCEL" in text
    has_lyocell = "LYOCELL" in text

    if text.startswith("PC"):
        return (
            "Polyester – Cotton Stretch"
            if stretch
            else "Polyester – Cotton Blend"
        )
    if text == "C/P(35/65)":
        return "Polyester – Cotton Blend"

    if (
        text.startswith("M/C/P/S")
        or (has_cotton and has_lyocell and has_polyester)
        or (has_polyester and has_bamboo and "RAYON" in text)
    ):
        return "Multi-Fibre Blend"

    if has_cotton:
        if has_linen:
            return "Cotton – Linen Blend"
        if has_modal:
            return "Cotton – Modal Blend"
        if has_tencel:
            return "Cotton – Tencel Blend"
        if has_lyocell:
            return "Cotton – Lyocell Blend"
        if has_viscose:
            return "Cotton – Viscose Blend"
        if has_nylon:
            return "Cotton – Nylon Stretch" if stretch else "Cotton – Nylon Blend"
        if has_polyester or "PE" in text:
            return (
                "Cotton – Polyester Stretch"
                if stretch
                else "Cotton – Polyester Blend"
            )
        if stretch:
            return "Cotton – Stretch"
        if "BLEND" in text:
            return "Cotton – Other Blend"
        return "Cotton – 100%"

    if re.match(r"^\d+P\+\d+V\+\d+LIN", compact):
        return "Polyester – Viscose Linen Blend"
    if text.startswith("PV LINEN"):
        return (
            "Polyester – Viscose Linen Stretch"
            if stretch
            else "Polyester – Viscose Linen Blend"
        )
    if text.startswith(("PVS", "P/V/ELA")):
        return "Polyester – Viscose Stretch"
    if text.startswith("PV") or (has_polyester and has_viscose):
        return (
            "Polyester – Viscose Stretch"
            if stretch
            else "Polyester – Viscose Blend"
        )

    if has_bamboo and has_polyester:
        return (
            "Polyester – Bamboo Stretch"
            if stretch
            else "Polyester – Bamboo Blend"
        )
    if (has_viscose or text.startswith("R/N")) and has_nylon:
        return (
            "Viscose – Nylon Stretch"
            if stretch
            else "Viscose – Nylon Blend"
        )
    if text == "100% VISCOSE":
        return "Viscose – 100%"

    if has_polyester:
        if has_modal:
            return "Polyester – Other Blend"
        return "Polyester – Stretch" if stretch else "Polyester – 100%"
    if has_nylon:
        return "Nylon – Stretch" if stretch else "Nylon – 100%"
    if has_tencel:
        return "Tencel – 100%"
    if has_lyocell:
        return "Lyocell – Blend"
    if text == "MODAL":
        return "Modal – 100%"
    return "Review Required"


def preprocess_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    source_identifier_column: str,
) -> tuple[list[dict[str, Any]], CleaningReport]:
    """Copy rows and normalize their product identifier and fabric."""

    cleaned_rows: list[dict[str, Any]] = []
    original_identifiers: list[str] = []
    cleaned_identifiers: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        original = str(row.get(source_identifier_column) or "")
        try:
            identifier = clean_product_identifier(
                row.get(source_identifier_column)
            )
        except PreprocessingError as exc:
            raise PreprocessingError(
                f"{dataset} row {row_number}: {exc}"
            ) from exc
        cleaned = dict(row)
        cleaned[source_identifier_column] = identifier
        cleaned["CAT4"] = standardize_fabric(row.get("CAT4"))
        cleaned_rows.append(cleaned)
        original_identifiers.append(original)
        cleaned_identifiers.append(identifier)

    unique_before = len(set(original_identifiers))
    unique_after = len(set(cleaned_identifiers))
    report = CleaningReport(
        dataset=dataset,
        identifier_column="product_id",
        rows=len(cleaned_rows),
        identifier_values_changed=sum(
            original != cleaned
            for original, cleaned in zip(
                original_identifiers,
                cleaned_identifiers,
                strict=True,
            )
        ),
        unique_identifiers_before=unique_before,
        unique_identifiers_after=unique_after,
        identifier_collisions_introduced=max(unique_before - unique_after, 0),
        review_required_fabric=sum(
            row["CAT4"] == "Review Required" for row in cleaned_rows
        ),
        unspecified_fabric=sum(
            row["CAT4"] == "Unspecified" for row in cleaned_rows
        ),
    )
    return cleaned_rows, report
