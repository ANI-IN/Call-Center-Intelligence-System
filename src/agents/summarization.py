from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import SummaryResult, TranscriptionResult

SYSTEM_PROMPT = (
    "You are a call center analyst. Analyze the following call "
    "transcript and produce a structured summary.\n\n"
    "Rules:\n"
    "- call_purpose: 1-2 sentences on why the customer called\n"
    "- key_discussion_points: bullet list of topics covered\n"
    "- action_items: specific next steps with owner "
    "(agent/customer/system) and deadline if mentioned\n"
    "- resolution_status: resolved, unresolved, or escalated\n"
    "- sentiment_trajectory: how customer sentiment changed "
    '(e.g., "Frustrated -> Satisfied")\n'
    "- entities: extract product names, account references, "
    "dates, monetary amounts\n\n"
    "Be factual. Only include information explicitly stated "
    "in the transcript. Do not infer or hallucinate."
)


class SummarizationError(Exception):
    pass


def _secs_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _format_transcript(transcript: TranscriptionResult) -> str:
    lines: list[str] = []
    for seg in transcript.segments:
        start = _secs_to_mmss(seg.start_time)
        end = _secs_to_mmss(seg.end_time)
        lines.append(f"[{start}-{end}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def run_summarization(
    transcript: TranscriptionResult,
    max_retries: int = 3,
    model: str = "gpt-4o",
    timeout: int = 30,
) -> SummaryResult:
    llm = ChatOpenAI(model=model, timeout=timeout)
    structured_llm = llm.with_structured_output(SummaryResult)

    formatted = _format_transcript(transcript)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Transcript:\n\n{formatted}"),
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

    raise SummarizationError(f"Failed after {max_retries} attempts. Last error: {last_error}")
