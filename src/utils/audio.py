from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from src.graph.state import AudioProperties

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
MAX_DURATION_SECONDS = 3600  # 60 minutes
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}


class AudioValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    is_valid: bool
    error: str | None = None


def detect_audio_format(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[:4] == b"fLaC":
        return "flac"
    if data[4:8] == b"ftyp":
        return "m4a"
    return None


def validate_audio_file(data: bytes, filename: str) -> ValidationResult:
    if len(data) == 0:
        return ValidationResult(is_valid=False, error="Empty file")

    if len(data) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            error=f"File size {len(data)} bytes exceeds maximum {MAX_FILE_SIZE_BYTES} bytes",
        )

    fmt = detect_audio_format(data)
    if fmt is None or fmt not in SUPPORTED_FORMATS:
        return ValidationResult(
            is_valid=False,
            error=f"Unsupported audio format for file '{filename}'. Supported: {SUPPORTED_FORMATS}",
        )

    return ValidationResult(is_valid=True)


def extract_audio_properties(data: bytes, fmt: str) -> AudioProperties:
    try:
        if fmt == "wav":
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                duration = frames / rate
            return AudioProperties(
                duration_seconds=round(duration, 2),
                sample_rate=rate,
                channels=channels,
                format=fmt,
                file_size_bytes=len(data),
            )
        # For non-wav formats, use mutagen or ffprobe in production.
        # Minimal fallback for mp3/flac/m4a:
        return AudioProperties(
            duration_seconds=0.0,
            sample_rate=0,
            channels=0,
            format=fmt,
            file_size_bytes=len(data),
        )
    except Exception as e:
        raise AudioValidationError(f"Failed to extract audio properties: {e}") from e
