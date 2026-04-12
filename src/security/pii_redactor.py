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
    (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE", "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> PIIRedactionResult:
    detections: list[PIIDetection] = []
    redacted = text

    for pattern, pii_type, replacement in PII_PATTERNS:
        for match in re.finditer(pattern, redacted):
            detections.append(
                PIIDetection(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
        redacted = re.sub(pattern, replacement, redacted)

    return PIIRedactionResult(
        redacted_text=redacted,
        pii_found=len(detections) > 0,
        detections=detections,
    )
