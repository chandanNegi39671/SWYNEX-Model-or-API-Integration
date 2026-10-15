"""PII redaction applied BEFORE any text is classified, logged or sent to an external API."""
import re

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d\b")

def _luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s += d; alt = not alt
    return s % 10 == 0

def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, sorted list of PII types found). Card numbers must pass the Luhn check."""
    found: set[str] = set()
    def sub(pattern, label, check=None):
        nonlocal text
        def r(m):
            if check and not check(m.group(0)): return m.group(0)
            found.add(label); return f"[{label}]"
        text = pattern.sub(r, text)
    sub(_IBAN, "IBAN")
    sub(_EMAIL, "EMAIL")
    sub(_CARD, "CARD_NUMBER", lambda s: _luhn(re.sub(r"\D", "", s)))
    sub(_PHONE, "PHONE")
    return text, sorted(found)
