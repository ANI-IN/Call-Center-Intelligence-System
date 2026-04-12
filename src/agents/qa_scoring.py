from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import (
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
)

SYSTEM_PROMPT = (
    "You are a call center quality assurance evaluator. "
    "Score the call agent's performance on 5 dimensions, "
    "each rated 1-5.\n\n"
    "Scoring rubric:\n"
    "- **Professionalism** (1-5): Appropriate language, "
    "no interruptions, proper greeting/closing\n"
    "- **Empathy** (1-5): Acknowledged customer feelings, "
    "active listening indicators\n"
    "- **Problem Resolution** (1-5): Identified root cause, "
    "provided solution, confirmed understanding\n"
    "- **Compliance** (1-5): Followed required disclosures, "
    "verification steps, hold procedures\n"
    "- **Communication Clarity** (1-5): Clear explanations, "
    "avoided jargon, confirmed understanding\n\n"
    "Rules:\n"
    "- Each score MUST include a justification citing "
    "specific transcript segments\n"
    "- ALL timestamps MUST be in MM:SS format "
    "(e.g., 01:45, 03:20). Never use raw seconds.\n"
    "- Calculate overall_score as weighted average: "
    "Professionalism 15%, Empathy 20%, "
    "Problem Resolution 30%, Compliance 20%, "
    "Communication Clarity 15%\n"
    "- Flag compliance violations separately with severity "
    "(low/medium/high/critical) and transcript_reference "
    "in MM:SS-MM:SS format\n"
    "- Be objective. Score based only on what is in "
    "the transcript."
)


class QAScoringError(Exception):
    pass


def _secs_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _format_input(transcript: TranscriptionResult, summary: SummaryResult) -> str:
    lines: list[str] = []
    lines.append("=== TRANSCRIPT ===")
    for seg in transcript.segments:
        start = _secs_to_mmss(seg.start_time)
        end = _secs_to_mmss(seg.end_time)
        lines.append(f"[{start}-{end}] {seg.speaker}: {seg.text}")
    lines.append("\n=== SUMMARY ===")
    lines.append(f"Purpose: {summary.call_purpose}")
    lines.append(f"Resolution: {summary.resolution_status.value}")
    lines.append(f"Sentiment: {summary.sentiment_trajectory}")
    return "\n".join(lines)


def run_qa_scoring(
    transcript: TranscriptionResult,
    summary: SummaryResult,
    max_retries: int = 3,
    model: str = "gpt-4o",
    timeout: int = 120,
) -> QAScoreResult:
    llm = ChatOpenAI(model=model, timeout=timeout)
    structured_llm = llm.with_structured_output(QAScoreResult)

    formatted = _format_input(transcript, summary)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=formatted),
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke(messages)
            result.call_id = transcript.call_id
            return result
        except Exception as e:
            last_error = e
            continue

    raise QAScoringError(f"Failed after {max_retries} attempts. Last error: {last_error}")
