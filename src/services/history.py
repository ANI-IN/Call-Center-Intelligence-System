"""History service — read-only access to persisted CallRecord rows.

The Analyze tab persists every successful call to ``CallRecord``. This service
exposes those rows for the History tab, with no schema change. Both functions
are pure: given an engine they return plain Python data structures suitable
for direct rendering by Gradio components.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.database.connection import session_scope
from src.database.models import CallRecord
from src.graph.state import CallReport
from src.utils.formatters import format_qa, format_summary

logger = logging.getLogger(__name__)


def _safe_loads(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return {}


def list_calls(engine, limit: int = 200) -> list[dict[str, Any]]:
    """Return up to ``limit`` past calls, newest-first.

    Each row is a dict with display-ready scalars only — no nested objects —
    so it can be handed directly to a ``gr.Dataframe``.
    """
    with session_scope(engine) as session:
        rows = (
            session.query(CallRecord)
            .order_by(CallRecord.processed_at.desc())
            .limit(limit)
            .all()
        )

        out: list[dict[str, Any]] = []
        for r in rows:
            qa = _safe_loads(r.qa_scores_json)
            summary = _safe_loads(r.summary_json)
            score = qa.get("overall_score")
            resolution = summary.get("resolution_status")
            out.append(
                {
                    "call_id": r.call_id,
                    "processed_at": (
                        r.processed_at.strftime("%Y-%m-%d %H:%M:%S")
                        if r.processed_at
                        else ""
                    ),
                    "audio_filename": r.audio_filename or "",
                    "status": r.status or "",
                    "overall_score": (
                        f"{float(score):.2f}/5" if isinstance(score, (int, float)) else "—"
                    ),
                    "resolution_status": str(resolution) if resolution else "—",
                }
            )
        return out


def get_call_detail(engine, call_id: str) -> dict[str, Any] | None:
    """Return the full analysis for one call, ready for the detail panel.

    Returns ``None`` if the call_id is unknown. The returned dict contains:

    - ``call_id``, ``processed_at``, ``audio_filename``, ``status``
    - ``transcript`` — the persisted transcript text
    - ``summary_md`` — markdown rendering of the summary
    - ``qa_md`` — markdown rendering of the QA scores
    - ``report`` — the rehydrated ``CallReport`` (or ``None`` if the stored
      JSON cannot be parsed; PDF/JSON downloads will be unavailable in that
      edge case but the rest of the panel still renders)
    """
    with session_scope(engine) as session:
        r: CallRecord | None = (
            session.query(CallRecord).filter_by(call_id=call_id).one_or_none()
        )
        if r is None:
            logger.info(f"get_call_detail: unknown call_id {call_id}")
            return None

        summary_dict = _safe_loads(r.summary_json)
        qa_dict = _safe_loads(r.qa_scores_json)

        report: CallReport | None = None
        if r.report_json:
            try:
                report = CallReport.model_validate_json(r.report_json)
            except Exception:
                logger.exception(f"get_call_detail: report_json invalid for {call_id}")

        return {
            "call_id": r.call_id,
            "processed_at": (
                r.processed_at.strftime("%Y-%m-%d %H:%M:%S") if r.processed_at else ""
            ),
            "audio_filename": r.audio_filename or "",
            "status": r.status or "",
            "transcript": r.transcript_text or "",
            "summary_md": (
                format_summary(summary_dict) if summary_dict else "_No summary available._"
            ),
            "qa_md": (
                format_qa(qa_dict) if qa_dict else "_No QA scores available._"
            ),
            "report": report,
        }
