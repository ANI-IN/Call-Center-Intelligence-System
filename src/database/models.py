from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(36), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False)
    audio_filename = Column(String(255), nullable=False)
    transcript_text = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)
    qa_scores_json = Column(Text, nullable=True)
    report_json = Column(Text, nullable=True)
    processed_at = Column(DateTime, nullable=False, default=datetime.now)
    trace_id = Column(String(255), nullable=True)


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(36), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    user = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    details = Column(Text, nullable=True)
