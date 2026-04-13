from __future__ import annotations

import hashlib
import logging
import re

from faster_whisper import WhisperModel

from src.graph.state import (
    IntakeResult,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

# Module-level singleton — load model ONCE
_model = None
_model_size = None


def _detect_device() -> tuple[str, str]:
    """Auto-detect best available device and compute type."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't work with CTranslate2, fall back to CPU
            return "cpu", "int8"
    except ImportError:
        pass
    return "cpu", "int8"


def _get_whisper_model(model_size: str = "base"):
    global _model, _model_size
    if _model is None or _model_size != model_size:
        device, compute_type = _detect_device()
        logger.info(f"Loading Whisper '{model_size}' on {device} ({compute_type})")
        _model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        _model_size = model_size
    return _model


# Patterns that strongly indicate agent speech (first speaker)
_AGENT_PATTERNS = re.compile(
    r"(?i)(thank you for calling|how (?:can|may) I (?:help|assist)|"
    r"my name is|this call (?:may|will) be|for quality (?:and|assurance)|"
    r"is there anything else|have a (?:great|good|wonderful) day)"
)
# Patterns that strongly indicate customer speech
_CUSTOMER_PATTERNS = re.compile(
    r"(?i)(I(?:'m| am) calling (?:about|because|to)|I (?:need|want|have a)|"
    r"my account|my order|my bill|can you help|I was charged)"
)


class SpeakerDiarizer:
    """Heuristic speaker detection using gaps, questions, and content patterns."""

    def assign_speakers(self, segments: list[dict]) -> list[str]:
        labels = ["Agent", "Customer"]
        assignments: list[str] = []
        current = 0  # 0 = Agent, 1 = Customer

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()

            if i == 0:
                # First segment: check if it sounds like an agent greeting
                if _AGENT_PATTERNS.search(text):
                    current = 0
                elif _CUSTOMER_PATTERNS.search(text):
                    current = 1
                # else default to Agent (call center convention)
            else:
                gap = seg["start"] - segments[i - 1]["end"]
                prev_text = segments[i - 1].get("text", "").strip()

                # Content-based: strong signal overrides gap heuristic
                if _AGENT_PATTERNS.search(text) and current != 0:
                    current = 0
                elif _CUSTOMER_PATTERNS.search(text) and current != 1:
                    current = 1
                # Gap-based: speaker likely changed
                elif gap > 1.2:
                    current = 1 - current
                # Question followed by answer = speaker change
                elif prev_text.endswith("?"):
                    current = 1 - current
                # Short affirmation after long segment = different speaker
                elif len(text.split()) <= 3 and len(prev_text.split()) > 10:
                    current = 1 - current

            assignments.append(labels[current])
        return assignments


_diarizer = SpeakerDiarizer()


def _get_diarizer() -> SpeakerDiarizer:
    return _diarizer


def _compute_audio_hash(audio_path: str) -> str:
    """Compute SHA-256 hash of an audio file for caching."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_cache(audio_hash: str, engine) -> TranscriptionResult | None:
    """Check if a transcription result exists in cache."""
    if engine is None:
        return None
    try:
        from src.database.connection import get_session
        from src.database.models import TranscriptionCache

        session = get_session(engine)
        try:
            cached = session.query(TranscriptionCache).filter_by(audio_hash=audio_hash).first()
            if cached:
                logger.info(f"Cache hit for audio hash {audio_hash[:12]}...")
                return TranscriptionResult.model_validate_json(cached.transcription_json)
            return None
        finally:
            session.close()
    except Exception:
        return None


def _save_cache(audio_hash: str, result: TranscriptionResult, engine) -> None:
    """Save transcription result to cache."""
    if engine is None:
        return
    try:
        from datetime import datetime

        from src.database.connection import get_session
        from src.database.models import TranscriptionCache

        session = get_session(engine)
        try:
            entry = TranscriptionCache(
                audio_hash=audio_hash,
                transcription_json=result.model_dump_json(),
                created_at=datetime.now(),
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()
    except Exception:
        pass  # Cache save failure is non-critical


# Common Whisper artifacts to clean
_ARTIFACT_PATTERNS = [
    re.compile(r"\[BLANK_AUDIO\]", re.IGNORECASE),
    re.compile(r"\(blank audio\)", re.IGNORECASE),
    re.compile(r"\.{4,}"),  # repeated dots (hallucination)
    re.compile(r"(?i)^(thanks for watching|subscribe|like and subscribe).*$"),  # YouTube artifacts
    re.compile(r"(?i)^(music|applause|laughter)\s*$"),  # non-speech tags
]


def _clean_transcript_text(text: str) -> str:
    """Remove common Whisper artifacts and normalize whitespace."""
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    # Collapse repeated phrases (e.g., "thank you thank you thank you" → "thank you")
    words = text.split()
    if len(words) >= 6:
        # Check if the same 1-3 word phrase repeats 3+ times
        for phrase_len in range(1, 4):
            if len(words) >= phrase_len * 3:
                phrase = " ".join(words[:phrase_len])
                repeats = 0
                for i in range(0, len(words), phrase_len):
                    if " ".join(words[i : i + phrase_len]) == phrase:
                        repeats += 1
                    else:
                        break
                if repeats >= 3:
                    text = phrase
                    break
    return text.strip()


def run_transcription(
    intake: IntakeResult,
    confidence_threshold: float = 0.3,
    halt_ratio: float = 0.8,
    model_size: str = "base",
    db_engine=None,
) -> TranscriptionResult:
    # Check cache first
    audio_hash = None
    try:
        audio_hash = _compute_audio_hash(intake.audio_path)
        cached = _check_cache(audio_hash, db_engine)
        if cached is not None:
            cached.call_id = intake.call_id
            return cached
    except (FileNotFoundError, OSError):
        pass  # File doesn't exist yet or can't be read — skip cache

    model = _get_whisper_model(model_size)

    # faster-whisper handles resampling and mono conversion internally
    # beam_size=1 (greedy) is ~2x faster than beam_size=5 with minimal quality loss
    # word_timestamps=True improves segment boundaries at negligible speed cost
    raw_segments, info = model.transcribe(
        intake.audio_path,
        beam_size=1,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},  # more precise speech boundaries
        word_timestamps=True,
        condition_on_previous_text=False,  # Prevents hallucination loops
    )

    # Collect segments with improved confidence and text cleanup
    seg_list = []
    full_text_parts = []
    for seg in raw_segments:
        text = _clean_transcript_text(seg.text.strip())
        if not text:
            continue  # skip empty/artifact segments

        # Better confidence: combine avg_logprob and no_speech_prob
        # avg_logprob ranges ~[-1, 0], no_speech_prob ranges [0, 1]
        logprob_conf = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
        speech_conf = 1.0 - seg.no_speech_prob
        conf = round(logprob_conf * 0.7 + speech_conf * 0.3, 4)

        seg_list.append(
            {
                "text": text,
                "start": seg.start,
                "end": seg.end,
                "confidence": conf,
            }
        )
        full_text_parts.append(text)

    # Assign speakers
    diarizer = _get_diarizer()
    speakers = diarizer.assign_speakers(seg_list)

    # Build typed segments
    segments: list[TranscriptionSegment] = []
    for i, s in enumerate(seg_list):
        segments.append(
            TranscriptionSegment(
                text=s["text"],
                start_time=s["start"],
                end_time=s["end"],
                speaker=(speakers[i] if i < len(speakers) else "Unknown"),
                confidence=s["confidence"],
                low_confidence=s["confidence"] < confidence_threshold,
            )
        )

    low_count = sum(1 for s in segments if s.low_confidence)
    total = len(segments) if segments else 1
    overall_conf = sum(s.confidence for s in segments) / total if segments else 0.0
    flagged = (low_count / total) > halt_ratio

    result = TranscriptionResult(
        call_id=intake.call_id,
        full_text=" ".join(full_text_parts),
        segments=segments,
        overall_confidence=round(overall_conf, 4),
        flagged_for_review=flagged,
    )

    # Save to cache for future identical audio
    if audio_hash:
        _save_cache(audio_hash, result, db_engine)

    return result
