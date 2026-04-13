import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agents.qa_scoring import QAScoringError, run_qa_scoring
from src.graph.state import (
    QADimensionScore,
    QAScoreResult,
)
from tests.conftest import make_summary, make_transcription


class TestRunQAScoring:
    @patch("src.agents.qa_scoring.get_llm")
    def test_returns_valid_qa_result(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
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

        call_id = uuid.uuid4()
        transcript = make_transcription(call_id=call_id)
        summary = make_summary(call_id=call_id)
        result = run_qa_scoring(transcript, summary)

        assert isinstance(result, QAScoreResult)
        assert result.call_id == transcript.call_id
        assert 1 <= result.overall_score <= 5

    @patch("src.agents.qa_scoring.get_llm")
    def test_raises_after_max_retries(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM failed")

        call_id = uuid.uuid4()
        transcript = make_transcription(call_id=call_id)
        summary = make_summary(call_id=call_id)
        with pytest.raises(QAScoringError, match="Failed after 3 attempts"):
            run_qa_scoring(transcript, summary, max_retries=3)


class TestScoreValidation:
    @patch("src.agents.qa_scoring.get_llm")
    def test_overall_score_recomputed_from_weights(self, mock_get_llm: MagicMock) -> None:
        """Even if LLM returns wrong overall_score, it gets recomputed."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.invoke.return_value = QAScoreResult(
            call_id=uuid.uuid4(),
            professionalism=QADimensionScore(score=5, justification="Perfect."),
            empathy=QADimensionScore(score=5, justification="Perfect."),
            problem_resolution=QADimensionScore(score=5, justification="Perfect."),
            compliance=QADimensionScore(score=5, justification="Perfect."),
            communication_clarity=QADimensionScore(score=5, justification="Perfect."),
            overall_score=3.0,  # LLM returned wrong score
            compliance_flags=[],
        )

        transcript = make_transcription()
        result = run_qa_scoring(transcript)

        # Should be recomputed: 5*0.15 + 5*0.20 + 5*0.30 + 5*0.20 + 5*0.15 = 5.0
        assert result.overall_score == 5.0
