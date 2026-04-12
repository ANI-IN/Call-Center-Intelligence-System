import io
import wave

import pytest

from src.utils.audio import (
    AudioValidationError,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DURATION = 3600  # 60 min


def _make_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    n_frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


class TestDetectAudioFormat:
    def test_wav_magic_bytes(self) -> None:
        wav_bytes = _make_wav_bytes()
        assert detect_audio_format(wav_bytes) == "wav"

    def test_mp3_magic_bytes(self) -> None:
        mp3_header = b"\xff\xfb\x90\x00" + b"\x00" * 100
        assert detect_audio_format(mp3_header) == "mp3"

    def test_flac_magic_bytes(self) -> None:
        flac_header = b"fLaC" + b"\x00" * 100
        assert detect_audio_format(flac_header) == "flac"

    def test_unknown_format(self) -> None:
        assert detect_audio_format(b"\x00\x00\x00\x00") is None


class TestValidateAudioFile:
    def test_valid_wav(self) -> None:
        wav_bytes = _make_wav_bytes(duration_seconds=5.0)
        result = validate_audio_file(wav_bytes, "test.wav")
        assert result.is_valid is True
        assert result.error is None

    def test_reject_unsupported_format(self) -> None:
        result = validate_audio_file(b"\x00\x00\x00\x00", "test.ogg")
        assert result.is_valid is False
        assert "Unsupported audio format" in result.error

    def test_reject_oversized_file(self) -> None:
        huge_bytes = b"\x00" * (MAX_FILE_SIZE + 1)
        result = validate_audio_file(huge_bytes, "huge.wav")
        assert result.is_valid is False
        assert "exceeds maximum" in result.error

    def test_reject_empty_file(self) -> None:
        result = validate_audio_file(b"", "empty.wav")
        assert result.is_valid is False


class TestExtractAudioProperties:
    def test_wav_properties(self) -> None:
        wav_bytes = _make_wav_bytes(duration_seconds=2.0, sample_rate=16000)
        props = extract_audio_properties(wav_bytes, "wav")
        assert props.format == "wav"
        assert props.sample_rate == 16000
        assert props.channels == 1
        assert 1.9 <= props.duration_seconds <= 2.1
        assert props.file_size_bytes == len(wav_bytes)

    def test_raises_for_corrupt_audio(self) -> None:
        with pytest.raises(AudioValidationError):
            extract_audio_properties(b"not audio data", "wav")
