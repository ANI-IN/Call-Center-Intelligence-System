# Architecture

## System Overview

LangGraph state machine with 6 nodes processing audio through: Intake -> Transcription -> Summarization -> QA Scoring -> Report.

## Design Decisions

1. **LangGraph state machine** -- explicit routing over magic orchestration. Each node is a pure function: typed input -> typed output.
2. **Pydantic contracts** -- every inter-node message is validated. No raw dicts.
3. **Whisper local inference** -- no API dependency for transcription. Model loaded once, reused across calls.
4. **Separate eval model** -- Claude judges GPT-4o outputs to avoid self-evaluation bias.
5. **SQLite + SQLCipher** -- single-file database with encryption. No external database dependency for local dev.

## Data Flow

```
AudioInput -> IntakeResult -> TranscriptionResult -> SummaryResult -> QAScoreResult -> CallReport
```

## Conditional Routing

- Invalid audio -> error (immediate halt)
- Low transcription confidence (>40% segments below threshold) -> flagged for review
- Malformed LLM output -> retry (up to 3x), then fail
- Critical compliance violation -> supervisor review queue
- Primary LLM timeout/failure -> fallback model

## Observability

Every node is decorated with `@traceable` (LangSmith). Full traces available per call.
