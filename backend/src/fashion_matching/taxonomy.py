"""Controlled fashion colour families used for strict catalogue retrieval."""

from __future__ import annotations

from typing import Any

# Families intentionally keep retail-relevant blue-green shades separate. This
# prevents AQUA, TEAL and GREEN garments from being admitted as generic BLUE.
COLOUR_FAMILIES = frozenset(
    {
        "BLACK",
        "WHITE",
        "GREY",
        "NEUTRAL",
        "BROWN",
        "RED",
        "ORANGE",
        "YELLOW",
        "GREEN",
        "TEAL",
        "AQUA",
        "BLUE",
        "PURPLE",
        "PINK",
        "METALLIC",
        "MULTI",
    }
)

_COLOUR_FAMILY_BY_NAME = {
    # Neutrals and earth colours
    "BEIGE": "NEUTRAL", "LIGHT BEIGE": "NEUTRAL", "CREAM": "NEUTRAL",
    "IVORY": "NEUTRAL", "OFF WHITE": "NEUTRAL", "ECRU": "NEUTRAL",
    "NATURAL": "NEUTRAL", "KHAKI": "NEUTRAL", "SAND": "NEUTRAL",
    "STONE": "NEUTRAL", "TAN": "NEUTRAL", "FAWN": "NEUTRAL",
    "BROWN": "BROWN", "LIGHT BROWN": "BROWN", "COFFEE": "BROWN",
    "CHOCOLATE": "BROWN", "COCOA": "BROWN",
    # Achromatics and metallics
    "BLACK": "BLACK", "WHITE": "WHITE", "GREY": "GREY", "GRAY": "GREY",
    "LIGHT GREY": "GREY", "DARK GREY": "GREY", "MEDIUM GREY": "GREY",
    "CHARCOAL": "GREY", "SILVER": "METALLIC", "GOLD": "METALLIC",
    "BRONZE": "METALLIC", "COPPER": "METALLIC", "METALLIC": "METALLIC",
    # Blues
    "BLUE": "BLUE", "LIGHT BLUE": "BLUE", "MEDIUM BLUE": "BLUE",
    "DARK BLUE": "BLUE", "SKY BLUE": "BLUE", "SKY": "BLUE",
    "LIGHT SKY": "BLUE", "NAVY": "BLUE", "NAVY BLUE": "BLUE",
    "ROYAL BLUE": "BLUE", "DUSTY BLUE": "BLUE", "STEEL": "BLUE",
    "INDIGO": "BLUE", "INDIGO BLUE": "BLUE", "DARK INDIGO": "BLUE",
    "LIGHT INDIGO": "BLUE", "DARK INDIGO BLUE": "BLUE", "CHARCOAL INDIGO": "BLUE",
    "PETROL BLUE": "BLUE",
    # Blue-green families remain separate from BLUE/GREEN.
    "AQUA": "AQUA", "CYAN": "AQUA", "TURQUOISE": "AQUA",
    "TEAL": "TEAL", "MUTED TEAL": "TEAL", "TEAL BLUE": "TEAL",
    "PETROL": "TEAL", "SEA GREEN": "TEAL",
    # Greens
    "GREEN": "GREEN", "LIGHT GREEN": "GREEN", "DARK GREEN": "GREEN",
    "FOREST GREEN": "GREEN", "DUSTY GREEN": "GREEN", "MINT": "GREEN",
    "MINT GREEN": "GREEN", "OLIVE": "GREEN", "DARK OLIVE": "GREEN",
    "LIGHT OLIVE": "GREEN", "LIME GREEN": "GREEN", "LT GREEN": "GREEN",
    "PISTA": "GREEN", "PISTA GREEN": "GREEN", "PIESTA GREEN": "GREEN",
    "SAGE": "GREEN", "SAGE GREEN": "GREEN", "BOTTLE GREEN": "GREEN",
    # Warm and red families
    "RED": "RED", "LIGHT RED": "RED", "MAROON": "RED", "WINE": "RED",
    "BURGUNDY": "RED", "RUST": "RED", "ORANGE": "ORANGE",
    "CORAL": "ORANGE", "PEACH": "ORANGE", "PALE PEACH": "ORANGE",
    "YELLOW": "YELLOW", "LIGHT YELLOW": "YELLOW", "LEMON": "YELLOW",
    "MUSTARD": "YELLOW", "OCHRE": "YELLOW",
    # Pink and purple
    "PINK": "PINK", "LIGHT PINK": "PINK", "ROSE": "PINK", "MAGENTA": "PINK",
    "MEGANTA": "PINK", "ONION": "PINK", "COCOA PINK": "PINK",
    "PURPLE": "PURPLE", "LIGHT PURPLE": "PURPLE", "VIOLET": "PURPLE",
    "LAVENDER": "PURPLE", "MAUVE": "PURPLE", "LIGHT MAUVE": "PURPLE",
    "MOUVE": "PURPLE",
    "MULTI": "MULTI", "MULTICOLOUR": "MULTI", "MULTI COLOUR": "MULTI",
}


def canonical_colour_name(value: Any) -> str:
    """Normalise retail colour text without inventing a missing colour."""

    return " ".join(str(value or "").upper().replace("-", " ").split())


def colour_family(value: Any) -> str | None:
    """Resolve a controlled family, returning None for blank/invalid labels."""

    name = canonical_colour_name(value)
    if not name or name == "-":
        return None
    return _COLOUR_FAMILY_BY_NAME.get(name)


def colour_taxonomy_audit(values: list[Any]) -> dict[str, list[str]]:
    """Report source labels that cannot safely participate in strict matching."""

    unresolved = sorted({canonical_colour_name(value) for value in values if colour_family(value) is None})
    return {"unresolved": unresolved}
