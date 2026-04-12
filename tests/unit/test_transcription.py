import uuid
from unittest.mock import MagicMock, patch

from src.agents.transcription import run_transcription
from src.graph.state import (
    AudioProperties,
    IntakeResult,
    PIIScanResult,
    TranscriptionResult,
)


def _make_intake_result(
    audio_path: str = "/tmp/test.wav",
) -> IntakeResult:
    return IntakeResult(
        call_id=uuid.uuid4(),
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


def _make_mock_segment(text, start, end, avg_logprob=-0.1):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.avg_logprob = avg_logprob
    return seg


class TestRunTranscription:
    @patch("src.agents.transcription._get_whisper_model")
    def test_successful_transcription(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        seg1 = _make_mock_segment("Hello.", 0.0, 1.0, -0.1)
        seg2 = _make_mock_segment("I have a billing issue.", 1.0, 3.0, -0.2)
        mock_info = MagicMock()
        mock_model.transcribe.return_value = (
            iter([seg1, seg2]),
            mock_info,
        )

        intake = _make_intake_result()
        result = run_transcription(intake, confidence_threshold=0.3, halt_ratio=0.8)

        assert isinstance(result, TranscriptionResult)
        assert result.call_id == intake.call_id
        assert len(result.segments) == 2
        assert result.flagged_for_review is False

    @patch("src.agents.transcription._get_whisper_model")
    def test_flags_low_confidence_call(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        seg1 = _make_mock_segment("mumble", 0.0, 1.0, -2.0)
        seg2 = _make_mock_segment("mumble", 1.0, 2.0, -2.5)
        mock_info = MagicMock()
        mock_model.transcribe.return_value = (
            iter([seg1, seg2]),
            mock_info,
        )

        intake = _make_intake_result()
        result = run_transcription(intake, confidence_threshold=0.3, halt_ratio=0.5)

        assert result.flagged_for_review is True
