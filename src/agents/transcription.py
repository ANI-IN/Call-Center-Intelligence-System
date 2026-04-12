from __future__ import annotations

from faster_whisper import WhisperModel

from src.graph.state import (
    IntakeResult,
    TranscriptionResult,
    TranscriptionSegment,
)

# Module-level singleton — load model ONCE
_model = None


def _get_whisper_model(model_size: str = "base"):
    global _model
    if _model is None:
        _model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
    return _model


class SpeakerDiarizer:
    """Heuristic speaker detection based on conversation gaps."""

    def assign_speakers(self, segments: list[dict]) -> list[str]:
        labels = ["Agent", "Customer"]
        assignments: list[str] = []
        current = 0
        for i, seg in enumerate(segments):
            if i > 0:
                gap = seg["start"] - segments[i - 1]["end"]
                prev = segments[i - 1].get("text", "").strip()
                if gap > 1.5 or prev.endswith("?"):
                    current = 1 - current
            assignments.append(labels[current])
        return assignments


_diarizer = SpeakerDiarizer()


def _get_diarizer() -> SpeakerDiarizer:
    return _diarizer


def run_transcription(
    intake: IntakeResult,
    confidence_threshold: float = 0.3,
    halt_ratio: float = 0.8,
    model_size: str = "base",
) -> TranscriptionResult:
    model = _get_whisper_model(model_size)

    # faster-whisper returns (segments_generator, info)
    raw_segments, info = model.transcribe(
        intake.audio_path,
        beam_size=5,
        language="en",
        vad_filter=True,
    )

    # Collect segments
    seg_list = []
    full_text_parts = []
    for seg in raw_segments:
        conf = round(max(0.0, min(1.0, seg.avg_logprob + 1.0)), 4)
        seg_list.append(
            {
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "confidence": conf,
            }
        )
        full_text_parts.append(seg.text.strip())

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

    return TranscriptionResult(
        call_id=intake.call_id,
        full_text=" ".join(full_text_parts),
        segments=segments,
        overall_confidence=round(overall_conf, 4),
        flagged_for_review=flagged,
    )
