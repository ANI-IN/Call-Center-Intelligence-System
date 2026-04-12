# src/evaluation/llm_judge.py
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for call center "
    "summarization systems. Given a transcript and a "
    "generated summary, evaluate the summary on "
    "4 dimensions.\n\n"
    "Score each dimension 1-5 with a written rationale:\n"
    "- **Factual Consistency**: Does the summary contain "
    "only claims supported by the transcript?\n"
    "- **Completeness**: Are all key discussion points "
    "from the transcript captured?\n"
    "- **Conciseness**: Is the summary free of unnecessary "
    "filler or repetition?\n"
    "- **Actionability**: Are action items specific, "
    "with clear ownership and deadlines?\n\n"
    "Be strict. A score of 5 means essentially perfect. "
    "A score of 3 means acceptable but with notable gaps."
)


class JudgeScore(BaseModel):
    factual_consistency: int = Field(ge=1, le=5)
    factual_consistency_rationale: str
    completeness: int = Field(ge=1, le=5)
    completeness_rationale: str
    conciseness: int = Field(ge=1, le=5)
    conciseness_rationale: str
    actionability: int = Field(ge=1, le=5)
    actionability_rationale: str


class LLMJudge:
    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        from langchain_anthropic import ChatAnthropic

        self.llm = ChatAnthropic(model=model)
        self.structured_llm = self.llm.with_structured_output(JudgeScore)

    def evaluate(self, transcript_text: str, summary_text: str) -> JudgeScore:
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"=== TRANSCRIPT ===\n{transcript_text}"
                    f"\n\n=== GENERATED SUMMARY ===\n"
                    f"{summary_text}"
                )
            ),
        ]
        return self.structured_llm.invoke(messages)
