# ruff: noqa: E402
from __future__ import annotations

import logging
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

logger = logging.getLogger(__name__)

from src.agents.intake import run_intake
from src.agents.qa_scoring import QAScoringError, run_qa_scoring
from src.agents.report import compile_report
from src.agents.summarization import SummarizationError, run_summarization
from src.agents.transcription import run_transcription
from src.graph.edges import route_after_intake, route_after_qa, route_after_transcription
from src.graph.state import (
    AudioInput,
    CallReport,
    IntakeResult,
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
)
from src.security.injection_detector import detect_injection
from src.security.pii_redactor import redact_pii
from src.utils.config import Config


class PipelineState(TypedDict, total=False):
    audio_input: AudioInput
    intake: IntakeResult
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    report: CallReport
    error: str
    status: str


@traceable(name="intake_node")
def intake_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    result = run_intake(state["audio_input"])
    logger.info(f"Intake completed in {time.time() - t0:.1f}s")
    return {"intake": result}


@traceable(name="transcription_node")
def transcription_node(state: PipelineState, config: Config, db_engine=None) -> PipelineState:
    t0 = time.time()
    result = run_transcription(
        state["intake"],
        confidence_threshold=config.confidence_threshold,
        halt_ratio=config.low_confidence_halt_ratio,
        model_size=config.whisper_model_size,
        db_engine=db_engine,
    )
    logger.info(f"Transcription completed in {time.time() - t0:.1f}s")
    return {"transcription": result}


@traceable(name="injection_check_node")
def injection_check_node(state: PipelineState) -> PipelineState:
    """Check transcript for prompt injection attempts before sending to LLM."""
    transcription = state["transcription"]
    result = detect_injection(transcription.full_text)
    if result.injection_detected:
        return {
            "status": "flagged_for_review",
            "error": (
                f"Prompt injection detected in transcript. "
                f"Patterns: {', '.join(result.matched_patterns)}"
            ),
        }
    return {}


@traceable(name="pii_redaction_node")
def pii_redaction_node(state: PipelineState) -> PipelineState:
    """Redact PII from transcript text before sending to LLM."""
    transcription = state["transcription"]

    # Redact full text
    full_result = redact_pii(transcription.full_text)
    redacted_full = full_result.redacted_text

    # Redact each segment
    redacted_segments = []
    for seg in transcription.segments:
        seg_result = redact_pii(seg.text)
        redacted_segments.append(seg.model_copy(update={"text": seg_result.redacted_text}))

    redacted_transcription = transcription.model_copy(
        update={"full_text": redacted_full, "segments": redacted_segments}
    )
    return {"transcription": redacted_transcription}


@traceable(name="summarize_and_qa_node")
def summarize_and_qa_node(state: PipelineState, config: Config) -> PipelineState:
    """Run summarization first, then QA scoring with summary context.

    QA gets better accuracy when it can see the summary (resolution status,
    sentiment). The sequential LLM calls add minimal overhead since both
    are single-call operations and the total is still faster than the
    transcription step.
    """
    t0 = time.time()
    transcript = state["transcription"]
    errors: list[str] = []

    # Step 1: Summarization
    summary_result = None
    try:
        summary_result = run_summarization(
            transcript,
            max_retries=config.max_retries_per_node,
            timeout=config.llm_timeout_seconds,
            provider=config.llm_provider,
        )
    except SummarizationError as e:
        errors.append(f"Summarization: {e}")

    # Step 2: QA scoring — now with summary context for better accuracy
    qa_result = None
    try:
        qa_result = run_qa_scoring(
            transcript,
            summary=summary_result,  # Pass summary for resolution/sentiment context
            max_retries=config.max_retries_per_node,
            timeout=config.llm_timeout_seconds,
            provider=config.llm_provider,
        )
    except QAScoringError as e:
        errors.append(f"QA Scoring: {e}")

    logger.info(f"Summarization + QA completed in {time.time() - t0:.1f}s")

    if errors:
        return {"error": "; ".join(errors), "status": "failed"}

    return {"summary": summary_result, "qa_scores": qa_result}


@traceable(name="report_node")
def report_node(state: PipelineState) -> PipelineState:
    report = compile_report(
        intake=state["intake"],
        transcription=state["transcription"],
        summary=state["summary"],
        qa_scores=state["qa_scores"],
        trace_id="",
    )
    return {"report": report, "status": "completed"}


def error_node(state: PipelineState) -> PipelineState:
    """Terminal node for failures. Surfaces the most specific error available.

    Failures originate in different places: intake stores its reason in
    ``intake.validation_error`` (the IntakeResult), while later nodes set
    ``state["error"]`` directly. The earlier shape was lost on the intake
    path, leaving users with a bare "Validation failed" message.
    """
    error = state.get("error")
    if not error:
        intake = state.get("intake")
        if intake is not None and getattr(intake, "validation_error", None):
            error = intake.validation_error
        else:
            error = "Validation failed (no detail captured)"
    logger.warning(f"Pipeline routed to error_step: {error}")
    return {"status": "failed", "error": error}


def supervisor_review_node(state: PipelineState) -> PipelineState:
    report = compile_report(
        intake=state["intake"],
        transcription=state["transcription"],
        summary=state["summary"],
        qa_scores=state["qa_scores"],
        trace_id="",
    )
    return {"report": report, "status": "flagged_for_review"}


def build_workflow(config: Config, db_engine=None) -> StateGraph:
    workflow = StateGraph(PipelineState)

    # Node names must NOT clash with PipelineState keys
    workflow.add_node("intake_step", intake_node)
    workflow.add_node("transcribe_step", lambda s: transcription_node(s, config, db_engine))
    workflow.add_node("injection_check_step", injection_check_node)
    workflow.add_node("pii_redact_step", pii_redaction_node)
    workflow.add_node("summarize_and_qa_step", lambda s: summarize_and_qa_node(s, config))
    workflow.add_node("report_step", report_node)
    workflow.add_node("error_step", error_node)
    workflow.add_node("supervisor_step", supervisor_review_node)

    # Set entry point
    workflow.set_entry_point("intake_step")

    # Conditional edges
    workflow.add_conditional_edges(
        "intake_step",
        lambda s: route_after_intake(s["intake"]),
        {"transcribe": "transcribe_step", "error": "error_step"},
    )
    workflow.add_conditional_edges(
        "transcribe_step",
        lambda s: route_after_transcription(s["transcription"]),
        {"summarize": "injection_check_step"},
    )
    # Injection check -> PII redaction -> parallel summarization + QA
    workflow.add_conditional_edges(
        "injection_check_step",
        lambda s: "error" if s.get("status") == "flagged_for_review" else "continue",
        {"error": "error_step", "continue": "pii_redact_step"},
    )
    workflow.add_edge("pii_redact_step", "summarize_and_qa_step")
    workflow.add_conditional_edges(
        "summarize_and_qa_step",
        lambda s: route_after_qa(s["qa_scores"]) if "qa_scores" in s else "error",
        {"report": "report_step", "supervisor_review": "supervisor_step", "error": "error_step"},
    )

    # Terminal edges
    workflow.add_edge("report_step", END)
    workflow.add_edge("error_step", END)
    workflow.add_edge("supervisor_step", END)

    return workflow


def compile_workflow(config: Config, db_engine=None):
    workflow = build_workflow(config, db_engine)
    return workflow.compile()
