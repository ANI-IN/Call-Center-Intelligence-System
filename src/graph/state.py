from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    audio_data: bytes
    filename: str
    caller_id: str | None = None
    timestamp: datetime | None = None
    department: str | None = None


class AudioProperties(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int


class PIIScanResult(BaseModel):
    pii_detected: bool
    redacted_fields: list[str]


class IntakeResult(BaseModel):
    call_id: uuid.UUID
    audio_path: str
    audio_properties: AudioProperties
    pii_scan: PIIScanResult
    validation_passed: bool
    validation_error: str | None


class TranscriptionSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool


class TranscriptionResult(BaseModel):
    call_id: uuid.UUID
    full_text: str
    segments: list[TranscriptionSegment]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    flagged_for_review: bool


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"


class ActionItem(BaseModel):
    description: str
    owner: Literal["agent", "customer", "system"]
    deadline: str | None = None


class Entity(BaseModel):
    text: str
    label: str


class SummaryResult(BaseModel):
    call_id: uuid.UUID
    call_purpose: str
    key_discussion_points: list[str]
    action_items: list[ActionItem]
    resolution_status: ResolutionStatus
    sentiment_trajectory: str
    entities: list[Entity]


class QADimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class ComplianceFlag(BaseModel):
    violation: str
    severity: Literal["low", "medium", "high", "critical"]
    transcript_reference: str


class QAScoreResult(BaseModel):
    call_id: uuid.UUID
    professionalism: QADimensionScore
    empathy: QADimensionScore
    problem_resolution: QADimensionScore
    compliance: QADimensionScore
    communication_clarity: QADimensionScore
    overall_score: float = Field(ge=1.0, le=5.0)
    compliance_flags: list[ComplianceFlag]


class CallReport(BaseModel):
    call_id: uuid.UUID
    intake: IntakeResult
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    processed_at: datetime
    trace_id: str
    status: Literal["completed", "failed", "flagged_for_review"]
