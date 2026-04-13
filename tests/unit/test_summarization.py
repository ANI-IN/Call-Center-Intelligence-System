import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarization import SummarizationError, run_summarization
from src.graph.state import (
    ResolutionStatus,
    SummaryResult,
)
from tests.conftest import make_transcription


class TestRunSummarization:
    @patch("src.agents.summarization.get_llm")
    def test_returns_valid_summary_result(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
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

        transcript = make_transcription()
        result = run_summarization(transcript)

        assert isinstance(result, SummaryResult)
        assert result.call_id == transcript.call_id
        assert result.resolution_status == ResolutionStatus.RESOLVED

    @patch("src.agents.summarization.get_llm")
    def test_retries_on_failure(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
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

        transcript = make_transcription()
        result = run_summarization(transcript, max_retries=3)
        assert isinstance(result, SummaryResult)

    @patch("src.agents.summarization.get_llm")
    def test_raises_after_max_retries(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM error")

        transcript = make_transcription()
        with pytest.raises(SummarizationError, match="Failed after 3 attempts"):
            run_summarization(transcript, max_retries=3)
