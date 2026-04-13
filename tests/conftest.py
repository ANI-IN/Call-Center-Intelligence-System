from __future__ import annotations

import io
import uuid
import wave

import pytest

from src.graph.state import (
    AudioProperties,
    IntakeResult,
    PIIScanResult,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def make_wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    n_frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def make_intake_result(
    audio_path: str = "/tmp/test.wav",
    call_id: uuid.UUID | None = None,
) -> IntakeResult:
    return IntakeResult(
        call_id=call_id or uuid.uuid4(),
        audio_path=audio_path,
        audio_properties=AudioProperties(
            duration_seconds=10.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=320000,
        ),
        pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
        validation_passed=True,
        validation_error=None,
    )


def make_transcription(
    text: str = "Hello. I have a billing issue.",
    call_id: uuid.UUID | None = None,
) -> TranscriptionResult:
    cid = call_id or uuid.uuid4()
    return TranscriptionResult(
        call_id=cid,
        full_text=text,
        segments=[
            TranscriptionSegment(
                text="Hello, how can I help you today?",
                start_time=0.0,
                end_time=2.0,
                speaker="Agent",
                confidence=0.95,
                low_confidence=False,
            ),
            TranscriptionSegment(
                text="I have a billing issue. I was charged $45.99 incorrectly.",
                start_time=2.0,
                end_time=6.0,
                speaker="Customer",
                confidence=0.92,
                low_confidence=False,
            ),
        ],
        overall_confidence=0.93,
        flagged_for_review=False,
    )


def make_summary(call_id: uuid.UUID | None = None) -> SummaryResult:
    return SummaryResult(
        call_id=call_id or uuid.uuid4(),
        call_purpose="Customer called about incorrect billing charge of $45.99.",
        key_discussion_points=["Billing dispute", "Incorrect charge"],
        action_items=[],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="Frustrated -> Satisfied",
        entities=[],
    )


def make_qa_scores(call_id: uuid.UUID | None = None) -> QAScoreResult:
    return QAScoreResult(
        call_id=call_id or uuid.uuid4(),
        professionalism=QADimensionScore(score=4, justification="Good greeting."),
        empathy=QADimensionScore(score=4, justification="Acknowledged."),
        problem_resolution=QADimensionScore(score=4, justification="Resolved."),
        compliance=QADimensionScore(score=4, justification="OK."),
        communication_clarity=QADimensionScore(score=4, justification="Clear."),
        overall_score=4.0,
        compliance_flags=[],
    )


@pytest.fixture
def db_engine(tmp_path):
    """Create a temporary SQLite engine for testing."""
    from src.database.connection import get_engine, init_db

    db_path = tmp_path / "test.db"
    engine = get_engine(str(db_path), encryption_key=None)
    init_db(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create a DB session from the temporary engine."""
    from src.database.connection import get_session

    session = get_session(db_engine)
    yield session
    session.close()
