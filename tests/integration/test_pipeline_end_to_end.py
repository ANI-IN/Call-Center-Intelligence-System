import io
import uuid
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import (
    AudioInput,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
)
from src.graph.workflow import compile_workflow
from src.utils.config import Config


def _make_test_config() -> Config:
    return Config(
        openai_api_key="test-key",
        langchain_api_key="test-key",
        langchain_project="test",
        anthropic_api_key="",
        db_encryption_key="test-key",
        db_path=Path("test.db"),
        gradio_username="test",
        gradio_password="test",
        max_cost_per_call_usd=2.0,
        max_retries_per_node=3,
        llm_timeout_seconds=30,
        whisper_model_size="base",
        confidence_threshold=0.6,
        low_confidence_halt_ratio=0.4,
    )


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


@pytest.mark.integration
class TestPipelineEndToEnd:
    @patch("src.agents.qa_scoring.ChatOpenAI")
    @patch("src.agents.summarization.ChatOpenAI")
    @patch("src.agents.transcription._get_whisper_model")
    def test_full_pipeline_happy_path(
        self,
        mock_get_model: MagicMock,
        mock_summary_llm: MagicMock,
        mock_qa_llm: MagicMock,
    ) -> None:
        # Mock faster-whisper model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        seg1 = MagicMock()
        seg1.text = "Hello."
        seg1.start = 0.0
        seg1.end = 1.0
        seg1.avg_logprob = -0.1
        seg2 = MagicMock()
        seg2.text = "I need help with billing."
        seg2.start = 1.0
        seg2.end = 3.0
        seg2.avg_logprob = -0.15
        mock_info = MagicMock()
        mock_model.transcribe.return_value = (
            iter([seg1, seg2]),
            mock_info,
        )

        # Mock summarization LLM
        mock_sum = MagicMock()
        mock_summary_llm.return_value = mock_sum
        mock_sum.with_structured_output.return_value = mock_sum
        mock_sum.invoke.return_value = SummaryResult(
            call_id=uuid.uuid4(),
            call_purpose="Customer needs billing help.",
            key_discussion_points=["Billing"],
            action_items=[],
            resolution_status=ResolutionStatus.RESOLVED,
            sentiment_trajectory="Neutral -> Satisfied",
            entities=[],
        )

        # Mock QA LLM
        mock_qa = MagicMock()
        mock_qa_llm.return_value = mock_qa
        mock_qa.with_structured_output.return_value = mock_qa
        mock_qa.invoke.return_value = QAScoreResult(
            call_id=uuid.uuid4(),
            professionalism=QADimensionScore(score=4, justification="Good greeting."),
            empathy=QADimensionScore(score=4, justification="Acknowledged."),
            problem_resolution=QADimensionScore(score=4, justification="Resolved."),
            compliance=QADimensionScore(score=4, justification="OK."),
            communication_clarity=QADimensionScore(score=4, justification="Clear."),
            overall_score=4.0,
            compliance_flags=[],
        )

        config = _make_test_config()
        app = compile_workflow(config)

        result = app.invoke(
            {"audio_input": AudioInput(audio_data=_make_wav_bytes(), filename="test.wav")}
        )

        assert result["status"] == "completed"
        assert result["report"] is not None

    def test_invalid_audio_routes_to_error(self) -> None:
        config = _make_test_config()
        app = compile_workflow(config)

        result = app.invoke(
            {"audio_input": AudioInput(audio_data=b"not audio", filename="bad.ogg")}
        )

        assert result["status"] == "failed"
