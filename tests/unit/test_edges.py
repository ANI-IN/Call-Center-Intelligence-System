import uuid

from src.graph.edges import route_after_intake, route_after_qa, route_after_transcription
from src.graph.state import (
    AudioProperties,
    ComplianceFlag,
    IntakeResult,
    PIIScanResult,
    TranscriptionResult,
)
from tests.conftest import make_intake_result, make_qa_scores


class TestRouteAfterIntake:
    def test_valid_intake_routes_to_transcribe(self) -> None:
        intake = make_intake_result()
        assert route_after_intake(intake) == "transcribe"

    def test_failed_intake_routes_to_error(self) -> None:
        intake = IntakeResult(
            call_id=uuid.uuid4(),
            audio_path="",
            audio_properties=AudioProperties(
                duration_seconds=0, sample_rate=0, channels=0, format="unknown", file_size_bytes=0
            ),
            pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
            validation_passed=False,
            validation_error="Bad format",
        )
        assert route_after_intake(intake) == "error"


class TestRouteAfterTranscription:
    def test_always_routes_to_summarize(self) -> None:
        transcript = TranscriptionResult(
            call_id=uuid.uuid4(),
            full_text="test",
            segments=[],
            overall_confidence=0.5,
            flagged_for_review=True,
        )
        assert route_after_transcription(transcript) == "summarize"


class TestRouteAfterQa:
    def test_normal_scores_route_to_report(self) -> None:
        qa = make_qa_scores()
        assert route_after_qa(qa) == "report"

    def test_critical_flag_routes_to_supervisor(self) -> None:
        qa = make_qa_scores()
        qa.compliance_flags = [
            ComplianceFlag(
                violation="Major issue", severity="critical", transcript_reference="01:00"
            )
        ]
        assert route_after_qa(qa) == "supervisor_review"

    def test_non_critical_flag_routes_to_report(self) -> None:
        qa = make_qa_scores()
        qa.compliance_flags = [
            ComplianceFlag(violation="Minor issue", severity="low", transcript_reference="02:00")
        ]
        assert route_after_qa(qa) == "report"
