"""Schema and post-cleaning validation gates."""

from __future__ import annotations

from typing import Any

from data_pipeline.preprocessing import (
    CONTROLLED_FABRIC_SET,
    IDENTIFIER_PATTERN,
)


class DataValidationError(ValueError):
    """Raised when a dataset fails a pipeline quality gate."""


def validate_cleaned_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    *,
    require_unique_identifiers: bool,
) -> None:
    invalid_identifiers = [
        row.get("product_id")
        for row in rows
        if not IDENTIFIER_PATTERN.fullmatch(str(row.get("product_id") or ""))
    ]
    if invalid_identifiers:
        raise DataValidationError(
            f"{dataset} contains {len(invalid_identifiers)} invalid identifiers; "
            f"examples: {invalid_identifiers[:5]}"
        )

    unknown_fabrics = sorted({
        str(row.get("fabric"))
        for row in rows
        if row.get("fabric") not in CONTROLLED_FABRIC_SET
    })
    if unknown_fabrics:
        raise DataValidationError(
            f"{dataset} contains fabric families outside the controlled vocabulary: "
            f"{unknown_fabrics[:5]}"
        )

    if require_unique_identifiers:
        identifiers = [str(row["product_id"]) for row in rows]
        duplicate_count = len(identifiers) - len(set(identifiers))
        if duplicate_count:
            raise DataValidationError(
                f"{dataset} contains {duplicate_count} duplicate identifiers"
            )
