from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base


def get_engine(db_path: str, encryption_key: str | None = None) -> Engine:
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Only attempt PRAGMA key if encryption_key is explicitly
    # provided AND not an auto-generated default (starts with check).
    # SQLCipher may not be available on all platforms.
    if encryption_key and os.environ.get("DB_ENCRYPTION_KEY"):
        try:

            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute(f"PRAGMA key='{encryption_key}'")
                cursor.close()

        except Exception:
            pass  # Fall back to unencrypted SQLite

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Session:
    session_factory = sessionmaker(bind=engine)
    return session_factory()
