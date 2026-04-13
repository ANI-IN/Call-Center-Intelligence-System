from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a call center analyst. Analyze the following call "
    "transcript and produce a structured summary.\n\n"
    "## Output Fields\n"
    "- **call_purpose**: 1-2 sentences on why the customer called. "
    "Start with the customer's core need, not the agent's greeting.\n"
    "- **key_discussion_points**: 3-7 bullet points of substantive topics. "
    "Do NOT include greetings, hold music, or pleasantries.\n"
    "- **action_items**: Specific next steps with owner "
    "(agent/customer/system) and deadline if mentioned. "
    "If no action items were discussed, return an empty list.\n"
    "- **resolution_status**: 'resolved' (issue fully addressed), "
    "'unresolved' (issue still open), or 'escalated' (transferred to another team)\n"
    "- **sentiment_trajectory**: How customer sentiment changed "
    "across the call. Use format 'Start -> End' "
    '(e.g., "Frustrated -> Satisfied", "Neutral -> Neutral")\n'
    "- **entities**: Extract product names, account numbers, "
    "dates, monetary amounts, and reference numbers mentioned.\n\n"
    "## Quality Rules\n"
    "- Be factual. Only include information EXPLICITLY stated in the transcript.\n"
    "- Do NOT infer, assume, or hallucinate details.\n"
    "- If parts of the transcript are unclear or garbled, acknowledge "
    "the limitation rather than guessing the content.\n"
    "- For very short calls (<30 seconds), it is acceptable to have "
    "fewer discussion points and entities."
)


class SummarizationError(Exception):
    pass


def _format_transcript(transcript: TranscriptionResult) -> str:
    lines: list[str] = []
    for seg in transcript.segments:
        start = secs_to_mmss(seg.start_time)
        end = secs_to_mmss(seg.end_time)
        lines.append(f"[{start}-{end}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def run_summarization(
    transcript: TranscriptionResult,
    max_retries: int = 3,
    model: str | None = None,
    timeout: int = 30,
    provider: str = "openai",
) -> SummaryResult:
    llm = get_llm(provider=provider, model=model, timeout=timeout)
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
            if attempt < max_retries - 1:
                wait = min(2**attempt, 10)
                logger.warning(f"Summarization attempt {attempt + 1} failed: {e}. Retry in {wait}s")
                time.sleep(wait)
            continue

    raise SummarizationError(f"Failed after {max_retries} attempts. Last error: {last_error}")
