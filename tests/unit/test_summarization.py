import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarization import SummarizationError, run_summarization
from src.graph.state import (
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_transcription(text: str = "Hello. I have a billing issue.") -> TranscriptionResult:
    return TranscriptionResult(
        call_id=uuid.uuid4(),
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


class TestRunSummarization:
    @patch("src.agents.summarization.ChatOpenAI")
    def test_returns_valid_summary_result(self, mock_chat_cls: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = SummaryResult(
            call_id=uuid.uuid4(),
            call_purpose="Customer called about incorrect billing charge of $45.99.",
            key_discussion_points=["Billing dispute", "Incorrect charge"],
            action_items=[],
            resolution_status=ResolutionStatus.RESOLVED,
            sentiment_trajectory="Frustrated -> Satisfied",
            entities=[],
        )

        transcript = _make_transcription()
        result = run_summarization(transcript)

        assert isinstance(result, SummaryResult)
        assert result.call_id == transcript.call_id
        assert result.resolution_status == ResolutionStatus.RESOLVED

    @patch("src.agents.summarization.ChatOpenAI")
    def test_retries_on_failure(self, mock_chat_cls: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            Exception("LLM error"),
            Exception("LLM error again"),
            SummaryResult(
                call_id=uuid.uuid4(),
                call_purpose="Test",
                key_discussion_points=[],
                action_items=[],
                resolution_status=ResolutionStatus.UNRESOLVED,
                sentiment_trajectory="Neutral",
                entities=[],
            ),
        ]

        transcript = _make_transcription()
        result = run_summarization(transcript, max_retries=3)
        assert isinstance(result, SummaryResult)

    @patch("src.agents.summarization.ChatOpenAI")
    def test_raises_after_max_retries(self, mock_chat_cls: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM error")

        transcript = _make_transcription()
        with pytest.raises(SummarizationError, match="Failed after 3 attempts"):
            run_summarization(transcript, max_retries=3)
