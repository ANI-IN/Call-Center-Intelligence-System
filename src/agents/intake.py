from __future__ import annotations

import io
import re
import tempfile
import uuid
import wave

from src.graph.state import AudioInput, AudioProperties, IntakeResult, PIIScanResult
from src.utils.audio import (
    MAX_DURATION_SECONDS,
    AudioValidationError,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)

# Simple regex patterns for PII in metadata fields
PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "CREDIT_CARD"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "EMAIL"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE"),
]

_EMPTY_AUDIO_PROPS = AudioProperties(
    duration_seconds=0, sample_rate=0, channels=0, format="unknown", file_size_bytes=0
)
_EMPTY_PII = PIIScanResult(pii_detected=False, redacted_fields=[])


def _scan_metadata_for_pii(audio_input: AudioInput) -> PIIScanResult:
    fields_to_scan = {
        "caller_id": audio_input.caller_id,
        "department": audio_input.department,
    }
    redacted: list[str] = []
    for field_name, value in fields_to_scan.items():
        if value is None:
            continue
        for pattern, pii_type in PII_PATTERNS:
            if re.search(pattern, value):
                redacted.append(f"{field_name}:{pii_type}")
    return PIIScanResult(
        pii_detected=len(redacted) > 0,
        redacted_fields=redacted,
    )


def _check_wav_duration(data: bytes) -> float | None:
    """Read duration from WAV header without loading full audio data.

    Returns duration in seconds, or None if the data cannot be parsed as WAV.
    """
    try:
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return None
            return frames / rate
    except Exception:
        return None


def _make_failed_result(call_id: uuid.UUID, error: str) -> IntakeResult:
    return IntakeResult(
        call_id=call_id,
        audio_path="",
        audio_properties=_EMPTY_AUDIO_PROPS,
        pii_scan=_EMPTY_PII,
        validation_passed=False,
        validation_error=error,
    )


def run_intake(audio_input: AudioInput) -> IntakeResult:
    call_id = uuid.uuid4()

    # Reject empty files immediately
    if len(audio_input.audio_data) == 0:
        return _make_failed_result(call_id, "Empty file")

    # For WAV files, check duration from header before the file-size gate so that
    # oversized-but-valid-format files get a "duration" error rather than a "file
    # size" error when the duration is the real constraint.
    fmt_early = detect_audio_format(audio_input.audio_data)
    if fmt_early == "wav":
        duration = _check_wav_duration(audio_input.audio_data)
        if duration is not None and duration > MAX_DURATION_SECONDS:
            return IntakeResult(
                call_id=call_id,
                audio_path="",
                audio_properties=AudioProperties(
                    duration_seconds=round(duration, 2),
                    sample_rate=0,
                    channels=0,
                    format="wav",
                    file_size_bytes=len(audio_input.audio_data),
                ),
                pii_scan=_EMPTY_PII,
                validation_passed=False,
                validation_error=(
                    f"Audio duration {round(duration, 2)}s exceeds maximum {MAX_DURATION_SECONDS}s"
                ),
            )

    # Validate format and file size
    validation = validate_audio_file(audio_input.audio_data, audio_input.filename)
    if not validation.is_valid:
        return _make_failed_result(call_id, validation.error)

    # Extract full audio properties
    fmt = detect_audio_format(audio_input.audio_data)
    try:
        props = extract_audio_properties(audio_input.audio_data, fmt)
    except AudioValidationError as exc:
        return _make_failed_result(call_id, str(exc))

    # Duration check for non-WAV formats (WAV already handled above)
    if props.duration_seconds > MAX_DURATION_SECONDS:
        return IntakeResult(
            call_id=call_id,
            audio_path="",
            audio_properties=props,
            pii_scan=_EMPTY_PII,
            validation_passed=False,
            validation_error=(
                f"Audio duration {props.duration_seconds}s exceeds maximum {MAX_DURATION_SECONDS}s"
            ),
        )

    # Save audio to a temp file for downstream agents
    suffix = f".{fmt}" if fmt else ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"call_{call_id}_")
    tmp.write(audio_input.audio_data)
    tmp.close()

    # PII scan on metadata fields
    pii_scan = _scan_metadata_for_pii(audio_input)

    return IntakeResult(
        call_id=call_id,
        audio_path=tmp.name,
        audio_properties=props,
        pii_scan=pii_scan,
        validation_passed=True,
        validation_error=None,
    )
