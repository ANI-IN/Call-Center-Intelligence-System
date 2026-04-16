"""Unit tests for the History service."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

from src.database.connection import get_session
from src.database.models import CallRecord
from src.services.history import get_call_detail, list_calls
from tests.conftest import (
    make_intake_result,
    make_qa_scores,
    make_summary,
    make_transcription,
)


def _insert_record(
    engine,
    *,
    call_id: str | None = None,
    processed_at: datetime | None = None,
    audio_filename: str = "call.mp3",
    status: str = "completed",
    transcript_text: str = "Hello.",
    summary_dict: dict | None = None,
    qa_dict: dict | None = None,
    report_json: str | None = None,
) -> str:
    cid = call_id or str(uuid.uuid4())
    summary_dict = summary_dict if summary_dict is not None else {
        "call_purpose": "Billing question",
        "key_discussion_points": ["Charge dispute"],
        "action_items": [],
        "resolution_status": "resolved",
        "sentiment_trajectory": "Frustrated -> Satisfied",
        "entities": [],
    }
    qa_dict = qa_dict if qa_dict is not None else {
        "overall_score": 4.25,
        "professionalism": {"score": 4, "justification": "ok"},
        "empathy": {"score": 4, "justification": "ok"},
        "problem_resolution": {"score": 5, "justification": "ok"},
        "compliance": {"score": 4, "justification": "ok"},
        "communication_clarity": {"score": 4, "justification": "ok"},
        "compliance_flags": [],
    }
    session = get_session(engine)
    try:
        rec = CallRecord(
            call_id=cid,
            status=status,
            audio_filename=audio_filename,
            transcript_text=transcript_text,
            summary_json=json.dumps(summary_dict),
            qa_scores_json=json.dumps(qa_dict),
            report_json=report_json,
            processed_at=processed_at or datetime.now(),
            trace_id="",
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()
    return cid


class TestListCalls:
    def test_returns_empty_when_no_records(self, db_engine) -> None:
        assert list_calls(db_engine) == []

    def test_returns_rows_newest_first(self, db_engine) -> None:
        now = datetime.now()
        _insert_record(db_engine, audio_filename="old.mp3", processed_at=now - timedelta(hours=2))
        _insert_record(db_engine, audio_filename="newest.mp3", processed_at=now)
        _insert_record(
            db_engine, audio_filename="middle.mp3", processed_at=now - timedelta(hours=1)
        )

        rows = list_calls(db_engine)

        assert [r["audio_filename"] for r in rows] == ["newest.mp3", "middle.mp3", "old.mp3"]

    def test_row_has_display_ready_fields(self, db_engine) -> None:
        _insert_record(db_engine, audio_filename="x.mp3")

        rows = list_calls(db_engine)

        assert len(rows) == 1
        r = rows[0]
        assert set(r.keys()) >= {
            "call_id",
            "processed_at",
            "audio_filename",
            "status",
            "overall_score",
            "resolution_status",
        }
        assert r["overall_score"] == "4.25/5"
        assert r["resolution_status"] == "resolved"

    def test_handles_missing_score_gracefully(self, db_engine) -> None:
        _insert_record(db_engine, qa_dict={}, summary_dict={})

        rows = list_calls(db_engine)

        assert rows[0]["overall_score"] == "—"
        assert rows[0]["resolution_status"] == "—"

    def test_respects_limit(self, db_engine) -> None:
        for i in range(5):
            _insert_record(db_engine, audio_filename=f"call{i}.mp3")

        rows = list_calls(db_engine, limit=3)

        assert len(rows) == 3


class TestGetCallDetail:
    def test_unknown_call_id_returns_none(self, db_engine) -> None:
        assert get_call_detail(db_engine, "does-not-exist") is None

    def test_returns_full_payload_for_known_call(self, db_engine) -> None:
        # Build a real CallReport so report_json round-trips
        from src.graph.state import CallReport

        intake = make_intake_result()
        cid = intake.call_id
        report = CallReport(
            call_id=cid,
            intake=intake,
            transcription=make_transcription(call_id=cid),
            summary=make_summary(call_id=cid),
            qa_scores=make_qa_scores(call_id=cid),
            processed_at=datetime.now(),
            trace_id="",
            status="completed",
        )
        _insert_record(
            db_engine,
            call_id=str(cid),
            transcript_text="Hello. I have a billing issue.",
            report_json=report.model_dump_json(),
        )

        detail = get_call_detail(db_engine, str(cid))

        assert detail is not None
        assert detail["call_id"] == str(cid)
        assert detail["transcript"] == "Hello. I have a billing issue."
        assert "Call Purpose" in detail["summary_md"]
        assert "Overall Quality Score" in detail["qa_md"]
        assert detail["report"] is not None
        assert detail["report"].call_id == cid

    def test_handles_corrupt_report_json(self, db_engine) -> None:
        cid = _insert_record(db_engine, report_json="{not valid json")
        detail = get_call_detail(db_engine, cid)
        assert detail is not None
        assert detail["report"] is None
        # Summary/QA still render from their separate JSON columns
        assert "Call Purpose" in detail["summary_md"]
