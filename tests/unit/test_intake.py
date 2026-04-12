import io
import wave
from datetime import datetime

from src.agents.intake import run_intake
from src.graph.state import AudioInput, IntakeResult


def _make_wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    n_frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


class TestRunIntake:
    def test_valid_wav_intake(self) -> None:
        audio_input = AudioInput(
            audio_data=_make_wav_bytes(5.0),
            filename="call_001.wav",
            caller_id="C-123",
            timestamp=datetime(2026, 4, 12),
            department="billing",
        )
        result = run_intake(audio_input)
        assert isinstance(result, IntakeResult)
        assert result.validation_passed is True
        assert result.validation_error is None
        assert result.audio_properties.format == "wav"
        assert result.audio_properties.duration_seconds > 0

    def test_reject_unsupported_format(self) -> None:
        audio_input = AudioInput(
            audio_data=b"\x00\x00\x00\x00" * 100,
            filename="call.ogg",
        )
        result = run_intake(audio_input)
        assert result.validation_passed is False
        assert "Unsupported" in result.validation_error

    def test_reject_empty_file(self) -> None:
        audio_input = AudioInput(audio_data=b"", filename="empty.wav")
        result = run_intake(audio_input)
        assert result.validation_passed is False

    def test_generates_unique_call_id(self) -> None:
        wav = _make_wav_bytes()
        r1 = run_intake(AudioInput(audio_data=wav, filename="a.wav"))
        r2 = run_intake(AudioInput(audio_data=wav, filename="b.wav"))
        assert r1.call_id != r2.call_id

    def test_pii_scan_on_metadata(self) -> None:
        audio_input = AudioInput(
            audio_data=_make_wav_bytes(),
            filename="call.wav",
            caller_id="SSN: 123-45-6789",
            department="billing",
        )
        result = run_intake(audio_input)
        assert result.pii_scan.pii_detected is True
        assert len(result.pii_scan.redacted_fields) > 0

    def test_reject_over_60_min_duration(self) -> None:
        audio_input = AudioInput(
            audio_data=_make_wav_bytes(duration_seconds=3601.0),
            filename="long_call.wav",
        )
        result = run_intake(audio_input)
        assert result.validation_passed is False
        assert "duration" in result.validation_error.lower()
