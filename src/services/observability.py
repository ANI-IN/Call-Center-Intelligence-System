"""Observability dashboard service — no UI dependencies."""

from __future__ import annotations

import json
import os

from src.database.connection import session_scope
from src.database.models import AuditLogEntry, CallRecord


def get_observability_dashboard(engine, langchain_project: str) -> tuple[str, str, list[list[str]]]:
    """Generate observability dashboard content.

    Returns (metrics_md, langsmith_md, audit_rows).
    """
    with session_scope(engine) as session:
        total_calls = session.query(CallRecord).count()
        completed = session.query(CallRecord).filter_by(status="completed").count()
        failed = session.query(CallRecord).filter_by(status="failed").count()
        flagged = session.query(CallRecord).filter_by(status="flagged_for_review").count()

        total_events = session.query(AuditLogEntry).count()

        recent_events = (
            session.query(AuditLogEntry).order_by(AuditLogEntry.timestamp.desc()).limit(20).all()
        )

        qa_rows = (
            session.query(CallRecord.qa_scores_json)
            .filter_by(status="completed")
            .filter(CallRecord.qa_scores_json.isnot(None))
            .all()
        )
        avg_score = 0.0
        score_count = 0
        compliance_flag_count = 0
        for (qa_json,) in qa_rows:
            try:
                q = json.loads(qa_json)
                s = q.get("overall_score")
                if s:
                    avg_score += float(s)
                    score_count += 1
                flags = q.get("compliance_flags", [])
                compliance_flag_count += len(flags)
            except (json.JSONDecodeError, TypeError):
                pass
        if score_count > 0:
            avg_score = round(avg_score / score_count, 2)

        lines = []
        lines.append("## Pipeline Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Total Calls Processed | **{total_calls}** |")
        lines.append(f"| Completed Successfully | **{completed}** |")
        lines.append(f"| Failed | **{failed}** |")
        lines.append(f"| Flagged for Review | **{flagged}** |")
        success_rate = f"{completed / total_calls * 100:.0f}%" if total_calls > 0 else "N/A"
        lines.append(f"| Success Rate | **{success_rate}** |")
        lines.append(f"| Avg Quality Score | **{avg_score}/5** |")
        lines.append(f"| Total Compliance Flags | **{compliance_flag_count}** |")
        lines.append(f"| Total Audit Events | **{total_events}** |")

        dashboard_md = "\n".join(lines)

        langsmith_url = "https://smith.langchain.com/o/default/projects"
        tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

        langsmith_md = "## LangSmith Integration\n\n"
        if tracing_enabled:
            langsmith_md += (
                f"Tracing is **enabled** for project: "
                f"`{langchain_project}`\n\n"
                f"[Open LangSmith Dashboard]({langsmith_url})\n\n"
                "Every pipeline run is traced with:\n"
                "- Input/output for each agent node\n"
                "- Latency per node\n"
                "- Token usage and cost per LLM call\n"
                "- Full conversation replay\n"
            )
        else:
            langsmith_md += (
                "Tracing is **disabled**. To enable:\n\n"
                "```\n"
                "LANGCHAIN_TRACING_V2=true\n"
                "LANGCHAIN_API_KEY=your-key\n"
                f"LANGCHAIN_PROJECT={langchain_project}\n"
                "```\n"
            )

        audit_rows = []
        for e in recent_events:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else ""
            details = ""
            if e.details:
                try:
                    d = json.loads(e.details)
                    details = str(d)[:60]
                except json.JSONDecodeError:
                    details = e.details[:60]
            audit_rows.append([ts, e.call_id[:12] + "...", e.action, details])

        return dashboard_md, langsmith_md, audit_rows
