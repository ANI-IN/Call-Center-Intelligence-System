import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agents.qa_scoring import QAScoringError, run_qa_scoring
from src.graph.state import (
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_transcript_and_summary() -> tuple[TranscriptionResult, SummaryResult]:
    call_id = uuid.uuid4()
    transcript = TranscriptionResult(
        call_id=call_id,
        full_text="Hello. I have a problem. Let me help you.",
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
                text="I have a billing problem.",
                start_time=2.0,
                end_time=4.0,
                speaker="Customer",
                confidence=0.92,
                low_confidence=False,
            ),
        ],
        overall_confidence=0.93,
        flagged_for_review=False,
    )
    summary = SummaryResult(
        call_id=call_id,
        call_purpose="Billing dispute",
        key_discussion_points=["Billing"],
        action_items=[],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="Neutral -> Satisfied",
        entities=[],
    )
    return transcript, summary


class TestRunQAScoring:
    @patch("src.agents.qa_scoring.ChatOpenAI")
    def test_returns_valid_qa_result(self, mock_chat_cls: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = QAScoreResult(
            call_id=uuid.uuid4(),
            professionalism=QADimensionScore(score=4, justification="Proper greeting at 0:00."),
            empathy=QADimensionScore(score=3, justification="Acknowledged issue."),
            problem_resolution=QADimensionScore(score=4, justification="Resolved."),
            compliance=QADimensionScore(score=5, justification="All steps followed."),
            communication_clarity=QADimensionScore(score=4, justification="Clear."),
            overall_score=4.0,
            compliance_flags=[],
        )

        transcript, summary = _make_transcript_and_summary()
        result = run_qa_scoring(transcript, summary)

        assert isinstance(result, QAScoreResult)
        assert result.call_id == transcript.call_id
        assert 1 <= result.overall_score <= 5

    @patch("src.agents.qa_scoring.ChatOpenAI")
    def test_raises_after_max_retries(self, mock_chat_cls: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM failed")

        transcript, summary = _make_transcript_and_summary()
        with pytest.raises(QAScoringError, match="Failed after 3 attempts"):
            run_qa_scoring(transcript, summary, max_retries=3)
