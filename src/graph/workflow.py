from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

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
    result = run_intake(state["audio_input"])
    return {"intake": result}


@traceable(name="transcription_node")
def transcription_node(state: PipelineState, config: Config) -> PipelineState:
    result = run_transcription(
        state["intake"],
        confidence_threshold=config.confidence_threshold,
        halt_ratio=config.low_confidence_halt_ratio,
        model_size=config.whisper_model_size,
    )
    return {"transcription": result}


@traceable(name="summarization_node")
def summarization_node(state: PipelineState, config: Config) -> PipelineState:
    try:
        result = run_summarization(
            state["transcription"],
            max_retries=config.max_retries_per_node,
            timeout=config.llm_timeout_seconds,
        )
        return {"summary": result}
    except SummarizationError as e:
        return {"error": str(e), "status": "summary_failed"}


@traceable(name="qa_scoring_node")
def qa_scoring_node(state: PipelineState, config: Config) -> PipelineState:
    try:
        result = run_qa_scoring(
            state["transcription"],
            state["summary"],
            max_retries=config.max_retries_per_node,
            timeout=config.llm_timeout_seconds,
        )
        return {"qa_scores": result}
    except QAScoringError as e:
        return {"error": str(e), "status": "qa_failed"}


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
    return {"status": "failed", "error": state.get("error", "Validation failed")}


def supervisor_review_node(state: PipelineState) -> PipelineState:
    report = compile_report(
        intake=state["intake"],
        transcription=state["transcription"],
        summary=state["summary"],
        qa_scores=state["qa_scores"],
        trace_id="",
    )
    return {"report": report, "status": "flagged_for_review"}


def build_workflow(config: Config) -> StateGraph:
    workflow = StateGraph(PipelineState)

    # Node names must NOT clash with PipelineState keys
    workflow.add_node("intake_step", intake_node)
    workflow.add_node("transcribe_step", lambda s: transcription_node(s, config))
    workflow.add_node("summarize_step", lambda s: summarization_node(s, config))
    workflow.add_node("qa_score_step", lambda s: qa_scoring_node(s, config))
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
        {"summarize": "summarize_step"},
    )
    workflow.add_edge("summarize_step", "qa_score_step")
    workflow.add_conditional_edges(
        "qa_score_step",
        lambda s: route_after_qa(s["qa_scores"]) if "qa_scores" in s else "error",
        {"report": "report_step", "supervisor_review": "supervisor_step", "error": "error_step"},
    )

    # Terminal edges
    workflow.add_edge("report_step", END)
    workflow.add_edge("error_step", END)
    workflow.add_edge("supervisor_step", END)

    return workflow


def compile_workflow(config: Config):
    workflow = build_workflow(config)
    return workflow.compile()
