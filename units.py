"""Extract numeric+unit mentions from text and normalize them to SI values per
dimension, so retrieval can match a query like "2.5 years" against a chunk
that says "30months" even though the embedding model may not treat those as
close in vector space.
"""
import re
from dataclasses import dataclass

# Each entry: unit pattern -> (dimension, factor to the dimension's SI unit)
_UNIT_FACTORS = {
    # length -> meters
    "km": ("length", 1000.0),
    "kilometers": ("length", 1000.0),
    "kilometres": ("length", 1000.0),
    "m": ("length", 1.0),
    "meters": ("length", 1.0),
    "metres": ("length", 1.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    # mass -> grams
    "kg": ("mass", 1000.0),
    "g": ("mass", 1.0),
    "mg": ("mass", 0.001),
    "mcg": ("mass", 1e-6),
    "µg": ("mass", 1e-6),
    # volume -> milliliters
    "l": ("volume", 1000.0),
    "ml": ("volume", 1.0),
    # time -> days
    "year": ("time", 365.0),
    "years": ("time", 365.0),
    "month": ("time", 30.0),
    "months": ("time", 30.0),
    "week": ("time", 7.0),
    "weeks": ("time", 7.0),
    "day": ("time", 1.0),
    "days": ("time", 1.0),
    "hour": ("time", 1 / 24),
    "hours": ("time", 1 / 24),
    # percent -> percent (dimensionless)
    "%": ("percent", 1.0),
    "percent": ("percent", 1.0),
    # temperature handled separately (needs offset, not just factor)
}

_NUMBER_UNIT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s?(?P<unit>km|kilometers|kilometres|kg|mcg|mg|ml|"
    r"years|year|months|month|weeks|week|hours|hour|days|day|cm|mm|"
    r"%|percent|[mgl])\b",
    re.IGNORECASE,
)

_TEMP_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s?(?:°\s?(?P<unit1>[cf])\b|degrees?\s?(?P<unit2>celsius|fahrenheit|c|f)\b)",
    re.IGNORECASE,
)


@dataclass
class NumericMention:
    raw: str
    value: float
    unit: str
    dimension: str
    si_value: float  # SI-per-dimension value (celsius for temperature)


def _normalize_temperature(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("f", "fahrenheit"):
        return (value - 32) * 5.0 / 9.0
    return value  # already celsius


def extract_numeric_mentions(text: str) -> list[NumericMention]:
    mentions: list[NumericMention] = []
    seen_spans = set()

    for m in _TEMP_RE.finditer(text):
        span = m.span()
        seen_spans.add(span)
        value = float(m.group("value"))
        unit = (m.group("unit1") or m.group("unit2") or "c").lower()
        celsius = _normalize_temperature(value, unit)
        mentions.append(NumericMention(m.group(0), value, unit, "temperature", celsius))

    for m in _NUMBER_UNIT_RE.finditer(text):
        span = m.span()
        if any(span[0] >= s[0] and span[1] <= s[1] for s in seen_spans):
            continue
        raw_unit = m.group("unit").lower().replace(" ", "")
        unit_key = raw_unit
        if unit_key not in _UNIT_FACTORS:
            continue
        dimension, factor = _UNIT_FACTORS[unit_key]
        value = float(m.group("value"))
        mentions.append(NumericMention(m.group(0), value, unit_key, dimension, value * factor))

    return mentions


def numeric_mentions_match(
    query_mentions: list[NumericMention],
    chunk_mentions: list[NumericMention],
    tolerance: float = 0.02,
) -> bool:
    """True if any query mention is within `tolerance` relative error of any
    chunk mention in the same dimension (e.g. 2.5 years ~= 30 months)."""
    for q in query_mentions:
        for c in chunk_mentions:
            if q.dimension != c.dimension:
                continue
            if c.si_value == 0:
                if q.si_value == 0:
                    return True
                continue
            rel_err = abs(q.si_value - c.si_value) / abs(c.si_value)
            if rel_err <= tolerance:
                return True
    return False
