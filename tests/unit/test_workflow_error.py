"""Tests for the terminal error_node in the LangGraph workflow.

Regression coverage for the bug where intake validation failures surfaced as
the bare string "Pipeline failed: Validation failed" with no detail, because
``error_node`` only read ``state["error"]`` and never inspected the
``IntakeResult.validation_error`` field where intake actually stores its
reason.
"""

from __future__ import annotations

import uuid

from src.graph.state import AudioProperties, IntakeResult, PIIScanResult
from src.graph.workflow import error_node


def _make_failed_intake(reason: str) -> IntakeResult:
    return IntakeResult(
        call_id=uuid.uuid4(),
        audio_path="",
        audio_properties=AudioProperties(
            duration_seconds=0,
            sample_rate=0,
            channels=0,
            format="unknown",
            file_size_bytes=0,
        ),
        pii_scan=PIIScanResult(pii_detected=False, redacted_fields=[]),
        validation_passed=False,
        validation_error=reason,
    )


class TestErrorNode:
    def test_uses_explicit_error_when_set(self) -> None:
        state = {"error": "Summarization: timeout"}
        out = error_node(state)
        assert out["status"] == "failed"
        assert out["error"] == "Summarization: timeout"

    def test_falls_back_to_intake_validation_error(self) -> None:
        # Regression: intake-failure path used to lose the actual reason.
        reason = "File size 60000000 bytes exceeds maximum 52428800 bytes"
        state = {"intake": _make_failed_intake(reason)}
        out = error_node(state)
        assert out["status"] == "failed"
        assert "exceeds maximum" in out["error"]

    def test_default_message_when_nothing_available(self) -> None:
        out = error_node({})
        assert out["status"] == "failed"
        assert out["error"] == "Validation failed (no detail captured)"

    def test_explicit_error_wins_over_intake_field(self) -> None:
        state = {
            "error": "Injection detected",
            "intake": _make_failed_intake("ignored"),
        }
        assert error_node(state)["error"] == "Injection detected"
