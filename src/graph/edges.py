from __future__ import annotations

from typing import Literal

from src.graph.state import IntakeResult, QAScoreResult, TranscriptionResult


def route_after_intake(intake: IntakeResult) -> Literal["transcribe", "error"]:
    if not intake.validation_passed:
        return "error"
    return "transcribe"


def route_after_transcription(
    transcription: TranscriptionResult,
) -> Literal["summarize"]:
    return "summarize"


def route_after_qa(qa: QAScoreResult) -> Literal["report", "supervisor_review"]:
    has_critical = any(f.severity == "critical" for f in qa.compliance_flags)
    if has_critical:
        return "supervisor_review"
    return "report"
