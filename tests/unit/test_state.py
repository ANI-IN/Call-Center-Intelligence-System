import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.graph.state import (
    ActionItem,
    AudioInput,
    AudioProperties,
    CallReport,
    ComplianceFlag,
    Entity,
    IntakeResult,
    PIIScanResult,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestAudioInput:
    def test_valid_audio_input(self) -> None:
        audio = AudioInput(
            audio_data=b"fake audio bytes",
            filename="call_001.wav",
            caller_id="C-12345",
            timestamp=datetime(2026, 4, 12, 10, 30, 0),
            department="billing",
        )
        assert audio.filename == "call_001.wav"
        assert audio.caller_id == "C-12345"

    def test_audio_input_optional_metadata(self) -> None:
        audio = AudioInput(audio_data=b"bytes", filename="call.wav")
        assert audio.caller_id is None
        assert audio.timestamp is None
        assert audio.department is None


class TestIntakeResult:
    def test_valid_intake_result(self) -> None:
        result = IntakeResult(
            call_id=uuid.uuid4(),
            audio_path="/tmp/call_001.wav",
            audio_properties=AudioProperties(
                duration_seconds=120.5,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=1024000,
            ),
            pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
            validation_passed=True,
            validation_error=None,
        )
        assert result.validation_passed is True

    def test_intake_result_with_validation_error(self) -> None:
        result = IntakeResult(
            call_id=uuid.uuid4(),
            audio_path="",
            audio_properties=AudioProperties(
                duration_seconds=0,
                sample_rate=0,
                channels=0,
                format="unknown",
                file_size_bytes=0,
            ),
            pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
            validation_passed=False,
            validation_error="Unsupported audio format: .ogg",
        )
        assert result.validation_passed is False
        assert result.validation_error is not None


class TestTranscriptionResult:
    def test_valid_transcription(self) -> None:
        result = TranscriptionResult(
            call_id=uuid.uuid4(),
            full_text="Hello, how can I help you today?",
            segments=[
                TranscriptionSegment(
                    text="Hello, how can I help you today?",
                    start_time=0.0,
                    end_time=2.5,
                    speaker="Agent",
                    confidence=0.95,
                    low_confidence=False,
                )
            ],
            overall_confidence=0.95,
            flagged_for_review=False,
        )
        assert result.overall_confidence == 0.95
        assert len(result.segments) == 1

    def test_low_confidence_segment(self) -> None:
        segment = TranscriptionSegment(
            text="mumble mumble",
            start_time=5.0,
            end_time=7.0,
            speaker="Customer",
            confidence=0.3,
            low_confidence=True,
        )
        assert segment.low_confidence is True

    def test_confidence_must_be_0_to_1(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionSegment(
                text="test",
                start_time=0.0,
                end_time=1.0,
                speaker="Agent",
                confidence=1.5,
                low_confidence=False,
            )


class TestSummaryResult:
    def test_valid_summary(self) -> None:
        result = SummaryResult(
            call_id=uuid.uuid4(),
            call_purpose="Customer called to dispute a charge on their billing statement.",
            key_discussion_points=["Billing dispute", "Charge reversal process"],
            action_items=[
                ActionItem(
                    description="Reverse charge of $45.99",
                    owner="agent",
                    deadline="2026-04-15",
                )
            ],
            resolution_status=ResolutionStatus.RESOLVED,
            sentiment_trajectory="Frustrated -> Satisfied",
            entities=[
                Entity(text="$45.99", label="AMOUNT"),
                Entity(text="April billing cycle", label="DATE"),
            ],
        )
        assert result.resolution_status == ResolutionStatus.RESOLVED
        assert len(result.action_items) == 1
        assert len(result.entities) == 2


class TestQAScoreResult:
    def test_valid_qa_scores(self) -> None:
        result = QAScoreResult(
            call_id=uuid.uuid4(),
            professionalism=QADimensionScore(
                score=4,
                justification=(
                    "Agent used proper greeting and maintained "
                    "professional tone. See segment 0:00-0:15."
                ),
            ),
            empathy=QADimensionScore(
                score=5,
                justification=(
                    "Agent acknowledged frustration: 'I understand this is frustrating' at 1:23."
                ),
            ),
            problem_resolution=QADimensionScore(
                score=4,
                justification=(
                    "Root cause identified and solution provided. "
                    "Confirmed customer understanding at 4:12."
                ),
            ),
            compliance=QADimensionScore(
                score=3,
                justification=(
                    "Verification completed but hold procedure not followed per standard at 2:30."
                ),
            ),
            communication_clarity=QADimensionScore(
                score=4,
                justification="Clear explanations given. Minimal jargon used.",
            ),
            overall_score=4.0,
            compliance_flags=[
                ComplianceFlag(
                    violation="Hold procedure not followed",
                    severity="medium",
                    transcript_reference="2:30-2:45",
                )
            ],
        )
        assert result.overall_score == 4.0
        assert len(result.compliance_flags) == 1

    def test_score_must_be_1_to_5(self) -> None:
        with pytest.raises(ValidationError):
            QADimensionScore(score=6, justification="Invalid score")

    def test_score_must_be_at_least_1(self) -> None:
        with pytest.raises(ValidationError):
            QADimensionScore(score=0, justification="Invalid score")


class TestCallReport:
    def test_valid_call_report(self) -> None:
        call_id = uuid.uuid4()
        report = CallReport(
            call_id=call_id,
            intake=IntakeResult(
                call_id=call_id,
                audio_path="/tmp/test.wav",
                audio_properties=AudioProperties(
                    duration_seconds=60.0,
                    sample_rate=16000,
                    channels=1,
                    format="wav",
                    file_size_bytes=500000,
                ),
                pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
                validation_passed=True,
                validation_error=None,
            ),
            transcription=TranscriptionResult(
                call_id=call_id,
                full_text="Hello",
                segments=[],
                overall_confidence=0.9,
                flagged_for_review=False,
            ),
            summary=SummaryResult(
                call_id=call_id,
                call_purpose="Test call",
                key_discussion_points=["Test"],
                action_items=[],
                resolution_status=ResolutionStatus.RESOLVED,
                sentiment_trajectory="Neutral",
                entities=[],
            ),
            qa_scores=QAScoreResult(
                call_id=call_id,
                professionalism=QADimensionScore(score=4, justification="Good"),
                empathy=QADimensionScore(score=4, justification="Good"),
                problem_resolution=QADimensionScore(score=4, justification="Good"),
                compliance=QADimensionScore(score=4, justification="Good"),
                communication_clarity=QADimensionScore(score=4, justification="Good"),
                overall_score=4.0,
                compliance_flags=[],
            ),
            processed_at=datetime.now(),
            trace_id="langsmith-trace-abc123",
            status="completed",
        )
        assert report.status == "completed"
