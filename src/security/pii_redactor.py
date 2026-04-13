from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIIDetection:
    pii_type: str
    original: str
    start: int
    end: int


@dataclass
class PIIRedactionResult:
    redacted_text: str
    pii_found: bool
    detections: list[PIIDetection] = field(default_factory=list)


PII_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", "SSN", "[REDACTED_SSN]"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "CREDIT_CARD", "[REDACTED_CREDIT_CARD]"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "EMAIL", "[REDACTED_EMAIL]"),
    (
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
        "PHONE",
        "[REDACTED_PHONE]",
    ),
]


def redact_pii(text: str) -> PIIRedactionResult:
    # Collect all matches from ORIGINAL text first to get correct positions
    all_matches: list[tuple[int, int, str, str, str]] = []

    for pattern, pii_type, replacement in PII_PATTERNS:
        for match in re.finditer(pattern, text):
            all_matches.append((match.start(), match.end(), pii_type, match.group(), replacement))

    if not all_matches:
        return PIIRedactionResult(redacted_text=text, pii_found=False)

    # Sort by start position descending so we replace from end to start
    # This preserves positions for earlier matches
    all_matches.sort(key=lambda m: m[0], reverse=True)

    # Deduplicate overlapping matches (keep the one that starts first)
    filtered: list[tuple[int, int, str, str, str]] = []
    for match in all_matches:
        if not filtered or match[1] <= filtered[-1][0]:
            filtered.append(match)
    filtered.reverse()

    detections: list[PIIDetection] = []
    redacted = text

    # Apply replacements from end to start to preserve positions
    for start, end, pii_type, original, replacement in reversed(filtered):
        detections.append(PIIDetection(pii_type=pii_type, original=original, start=start, end=end))
        redacted = redacted[:start] + replacement + redacted[end:]

    # Reverse detections so they're in order of appearance
    detections.reverse()

    return PIIRedactionResult(
        redacted_text=redacted,
        pii_found=True,
        detections=detections,
    )
