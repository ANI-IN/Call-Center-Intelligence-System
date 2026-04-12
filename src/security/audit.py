from __future__ import annotations

import json
from datetime import datetime

from src.database.connection import get_engine, get_session, init_db
from src.database.models import AuditLogEntry


class AuditLogger:
    def __init__(self, db_path: str, encryption_key: str | None = None) -> None:
        self.engine = get_engine(db_path, encryption_key)
        init_db(self.engine)

    def log(self, call_id: str, action: str, user: str, details: dict | None = None) -> None:
        session = get_session(self.engine)
        try:
            entry = AuditLogEntry(
                call_id=call_id,
                action=action,
                user=user,
                timestamp=datetime.now(),
                details=json.dumps(details) if details else None,
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

    def get_call_history(self, call_id: str) -> list[dict]:
        session = get_session(self.engine)
        try:
            entries = (
                session.query(AuditLogEntry)
                .filter_by(call_id=call_id)
                .order_by(AuditLogEntry.timestamp)
                .all()
            )
            return [
                {
                    "call_id": e.call_id,
                    "action": e.action,
                    "user": e.user,
                    "timestamp": e.timestamp.isoformat(),
                    "details": json.loads(e.details) if e.details else None,
                }
                for e in entries
            ]
        finally:
            session.close()
