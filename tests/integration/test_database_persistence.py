import uuid
from datetime import datetime

import pytest

from src.database.models import AuditLogEntry, CallRecord


@pytest.mark.integration
class TestCallRecordPersistence:
    def test_insert_and_retrieve_call_record(self, db_session) -> None:
        call_id = uuid.uuid4()
        record = CallRecord(
            call_id=str(call_id),
            status="completed",
            audio_filename="test.wav",
            transcript_text="Hello, how can I help?",
            summary_json='{"call_purpose": "test"}',
            qa_scores_json='{"overall_score": 4.0}',
            processed_at=datetime.now(),
            trace_id="trace-abc",
        )
        db_session.add(record)
        db_session.commit()

        retrieved = db_session.query(CallRecord).filter_by(call_id=str(call_id)).first()
        assert retrieved is not None
        assert retrieved.status == "completed"
        assert retrieved.transcript_text == "Hello, how can I help?"

    def test_insert_audit_log(self, db_session) -> None:
        entry = AuditLogEntry(
            call_id=str(uuid.uuid4()),
            action="pipeline_started",
            user="admin",
            timestamp=datetime.now(),
            details='{"node": "intake"}',
        )
        db_session.add(entry)
        db_session.commit()

        logs = db_session.query(AuditLogEntry).all()
        assert len(logs) == 1
        assert logs[0].action == "pipeline_started"
