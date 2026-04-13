from unittest.mock import MagicMock, patch

from src.agents.transcription import run_transcription
from src.graph.state import TranscriptionResult
from tests.conftest import make_intake_result


def _make_mock_segment(text, start, end, avg_logprob=-0.1, no_speech_prob=0.05):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.avg_logprob = avg_logprob
    seg.no_speech_prob = no_speech_prob
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

        intake = make_intake_result()
        result = run_transcription(intake, confidence_threshold=0.3, halt_ratio=0.8)

        assert isinstance(result, TranscriptionResult)
        assert result.call_id == intake.call_id
        assert len(result.segments) == 2
        assert result.flagged_for_review is False

    @patch("src.agents.transcription._get_whisper_model")
    def test_flags_low_confidence_call(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        seg1 = _make_mock_segment("mumble", 0.0, 1.0, -2.0, no_speech_prob=0.8)
        seg2 = _make_mock_segment("mumble", 1.0, 2.0, -2.5, no_speech_prob=0.9)
        mock_info = MagicMock()
        mock_model.transcribe.return_value = (
            iter([seg1, seg2]),
            mock_info,
        )

        intake = make_intake_result()
        result = run_transcription(intake, confidence_threshold=0.3, halt_ratio=0.5)

        assert result.flagged_for_review is True
