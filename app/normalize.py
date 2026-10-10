import re
import unicodedata


def normalize(t: str) -> str:
    """Case, punctuation, whitespace and unicode-insensitive form used for exact matching."""
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s°]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def numbers(t: str) -> list[str]:
    """Numeric literals in a question. Different given values mean a different question."""
    return sorted(set(re.findall(r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", t))))
