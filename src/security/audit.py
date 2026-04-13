from __future__ import annotations

import json
from datetime import datetime

from src.database.connection import get_engine, init_db, session_scope
from src.database.models import AuditLogEntry


class AuditLogger:
    def __init__(
        self,
        db_path: str | None = None,
        encryption_key: str | None = None,
        engine=None,
    ) -> None:
        if engine is not None:
            self.engine = engine
        else:
            self.engine = get_engine(db_path, encryption_key)
            init_db(self.engine)

    def log(self, call_id: str, action: str, user: str, details: dict | None = None) -> None:
        with session_scope(self.engine) as session:
            entry = AuditLogEntry(
                call_id=call_id,
                action=action,
                user=user,
                timestamp=datetime.now(),
                details=json.dumps(details) if details else None,
            )
            session.add(entry)
