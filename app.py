# app.py
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from src.agents.report import (
    generate_report_json,
    generate_report_pdf,
    persist_report,
)
from src.agents.transcription import _get_whisper_model
from src.database.connection import get_engine, get_session, init_db
from src.database.models import AuditLogEntry, CallRecord
from src.graph.state import AudioInput
from src.graph.workflow import compile_workflow
from src.security.audit import AuditLogger
from src.utils.config import load_config

# --- Startup: load everything ONCE ---
config = load_config()

_engine = get_engine(str(config.db_path), config.db_encryption_key)
init_db(_engine)

print(f"Loading Whisper model ({config.whisper_model_size})...")
_get_whisper_model(config.whisper_model_size)
print("Whisper model loaded.")

_workflow = compile_workflow(config)
_audit = AuditLogger(str(config.db_path), config.db_encryption_key)
print("Pipeline ready.")


# --- Formatting ---
def _format_summary(d: dict) -> str:
    lines = []
    lines.append(f"### Call Purpose\n{d.get('call_purpose', 'N/A')}\n")

    pts = d.get("key_discussion_points", [])
    if pts:
        lines.append("### Key Discussion Points")
        for p in pts:
            lines.append(f"- {p}")
        lines.append("")

    items = d.get("action_items", [])
    if items:
        lines.append("### Action Items")
        for it in items:
            owner = it.get("owner", "unknown").title()
            desc = it.get("description", "")
            dl = f" *(by {it['deadline']})*" if it.get("deadline") else ""
            lines.append(f"- **{owner}:** {desc}{dl}")
        lines.append("")

    status = d.get("resolution_status", "N/A")
    sentiment = d.get("sentiment_trajectory", "N/A")
    lines.append(f"### Resolution: `{status}`")
    lines.append(f"### Customer Sentiment: `{sentiment}`")

    entities = d.get("entities", [])
    if entities:
        lines.append("\n### Entities Mentioned")
        for e in entities:
            lines.append(f"- **{e.get('text', '')}** ({e.get('label', '')})")

    return "\n".join(lines)


def _format_qa(d: dict) -> str:
    lines = []
    overall = d.get("overall_score", 0)
    lines.append(f"### Overall Quality Score: {overall}/5\n")

    dims = [
        ("professionalism", "Professionalism"),
        ("empathy", "Empathy"),
        ("problem_resolution", "Problem Resolution"),
        ("compliance", "Compliance"),
        ("communication_clarity", "Communication Clarity"),
    ]
    for key, label in dims:
        dim = d.get(key, {})
        if isinstance(dim, dict):
            score = dim.get("score", "?")
            just = dim.get("justification", "")
            lines.append(f"**{label}: {score}/5**")
            if just:
                lines.append(f"> {just}")
            lines.append("")

    flags = d.get("compliance_flags", [])
    if flags:
        lines.append("### Compliance Flags")
        for f in flags:
            sev = f.get("severity", "").upper()
            viol = f.get("violation", "")
            ref = f.get("transcript_reference", "")
            lines.append(f"- **[{sev}]** {viol} *(ref: {ref})*")

    return "\n".join(lines)


def _format_history_detail(record: CallRecord) -> str:
    """Format a call record into readable markdown."""
    lines = []
    lines.append(f"**Call ID:** `{record.call_id}`")
    lines.append(f"**Status:** `{record.status}`")
    if record.processed_at:
        lines.append(f"**Processed:** {record.processed_at}")
    lines.append(f"**Audio:** {record.audio_filename}")
    lines.append("")

    if record.summary_json:
        try:
            s = json.loads(record.summary_json)
            lines.append("---")
            lines.append(f"**Purpose:** {s.get('call_purpose', '')}")
            res = s.get("resolution_status", "")
            sent = s.get("sentiment_trajectory", "")
            lines.append(f"**Resolution:** `{res}` | **Sentiment:** `{sent}`")
            pts = s.get("key_discussion_points", [])
            if pts:
                lines.append("**Key Points:**")
                for p in pts:
                    lines.append(f"  - {p}")
        except json.JSONDecodeError:
            pass

    if record.qa_scores_json:
        try:
            q = json.loads(record.qa_scores_json)
            overall = q.get("overall_score", "?")
            lines.append(f"\n**Quality Score:** {overall}/5")
            flags = q.get("compliance_flags", [])
            if flags:
                lines.append(f"**Compliance Flags:** {len(flags)} found")
        except json.JSONDecodeError:
            pass

    return "\n".join(lines)


# --- Pipeline ---
def process_call(audio_file, caller_id, department):
    """Process audio through the full pipeline."""
    if audio_file is None:
        return "No file uploaded.", "", "", None, None

    # Gradio type="numpy" returns (sample_rate, numpy_array)
    if isinstance(audio_file, tuple):
        import numpy as np
        import soundfile as sf

        sr, arr = audio_file
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            return "Empty audio recording.", "", "", None, None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, arr, sr)
        audio_file = tmp.name

    audio_path = Path(str(audio_file))
    if not audio_path.exists():
        return (
            f"File not found: {audio_file}",
            "",
            "",
            None,
            None,
        )

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    audio_input = AudioInput(
        audio_data=audio_data,
        filename=audio_path.name,
        caller_id=caller_id or None,
        department=department or None,
        timestamp=datetime.now(),
    )

    try:
        result = _workflow.invoke({"audio_input": audio_input})
    except Exception as e:
        return f"Pipeline error: {e}", "", "", None, None

    status = result.get("status", "unknown")

    if status == "failed":
        error = result.get("error", "Unknown error")
        _audit.log(
            call_id="unknown",
            action="pipeline_failed",
            user="app",
            details={"error": error},
        )
        return f"Pipeline failed: {error}", "", "", None, None

    report = result.get("report")
    if report is None:
        return "No report generated.", "", "", None, None

    persist_report(report, str(config.db_path), config.db_encryption_key)
    _audit.log(
        call_id=str(report.call_id),
        action="completed",
        user="app",
        details={"status": status},
    )

    # Transcript
    lines = []
    for seg in report.transcription.segments:
        tag = " [LOW CONF]" if seg.low_confidence else ""
        m = int(seg.start_time // 60)
        s = int(seg.start_time % 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg.speaker}: {seg.text}{tag}")
    transcript = "\n".join(lines)

    summary = _format_summary(report.summary.model_dump())
    qa = _format_qa(report.qa_scores.model_dump())

    json_path = tempfile.mktemp(suffix=".json")
    with open(json_path, "w") as f:
        f.write(generate_report_json(report))

    pdf_path = tempfile.mktemp(suffix=".pdf")
    with open(pdf_path, "wb") as f:
        f.write(generate_report_pdf(report))

    return transcript, summary, qa, json_path, pdf_path


def process_batch(audio_files):
    if not audio_files:
        return "No files uploaded."
    results = []
    for af in audio_files:
        t, *_ = process_call(af.name, None, None)
        results.append(f"--- {Path(af.name).name} ---\n{t or 'Failed'}\n")
    return "\n".join(results)


# --- History ---
def load_all_history():
    """Load all processed calls for history view."""
    session = get_session(_engine)
    try:
        records = (
            session.query(CallRecord).order_by(CallRecord.processed_at.desc()).limit(100).all()
        )
        if not records:
            return "No calls processed yet.", []

        table_data = []
        for r in records:
            # Table row
            processed = r.processed_at.strftime("%Y-%m-%d %H:%M") if r.processed_at else ""
            # Extract summary snippet
            purpose = ""
            overall = ""
            resolution = ""
            if r.summary_json:
                try:
                    s = json.loads(r.summary_json)
                    purpose = s.get("call_purpose", "")[:80]
                    resolution = s.get("resolution_status", "")
                except json.JSONDecodeError:
                    pass
            if r.qa_scores_json:
                try:
                    q = json.loads(r.qa_scores_json)
                    overall = str(q.get("overall_score", ""))
                except json.JSONDecodeError:
                    pass

            table_data.append(
                [
                    r.call_id[:12] + "...",
                    r.status,
                    processed,
                    resolution,
                    f"{overall}/5" if overall else "",
                    purpose,
                ]
            )

        summary_text = f"**{len(records)} calls processed**"
        return summary_text, table_data
    finally:
        session.close()


def get_call_detail(call_id):
    """Get formatted details for a specific call."""
    if not call_id or not call_id.strip():
        return "Select a call to view details.", "", ""

    # Handle truncated IDs from table
    search_id = call_id.strip().replace("...", "")

    session = get_session(_engine)
    try:
        if len(search_id) < 36:
            rec = session.query(CallRecord).filter(CallRecord.call_id.contains(search_id)).first()
        else:
            rec = session.query(CallRecord).filter_by(call_id=search_id).first()
        if not rec:
            return "Call not found.", "", ""

        # Format transcript with timestamps
        transcript = rec.transcript_text or "No transcript available."

        # Format summary
        summary_md = ""
        if rec.summary_json:
            try:
                summary_md = _format_summary(json.loads(rec.summary_json))
            except json.JSONDecodeError:
                summary_md = rec.summary_json

        # Format QA
        qa_md = ""
        if rec.qa_scores_json:
            try:
                qa_md = _format_qa(json.loads(rec.qa_scores_json))
            except json.JSONDecodeError:
                qa_md = rec.qa_scores_json

        return transcript, summary_md, qa_md
    finally:
        session.close()


# --- Observability ---
def get_observability_dashboard():
    """Generate observability dashboard content."""
    session = get_session(_engine)
    try:
        # Pipeline stats
        total_calls = session.query(CallRecord).count()
        completed = session.query(CallRecord).filter_by(status="completed").count()
        failed = session.query(CallRecord).filter_by(status="failed").count()
        flagged = session.query(CallRecord).filter_by(status="flagged_for_review").count()

        # Audit log stats
        total_events = session.query(AuditLogEntry).count()

        # Recent audit events
        recent_events = (
            session.query(AuditLogEntry).order_by(AuditLogEntry.timestamp.desc()).limit(20).all()
        )

        # Average QA scores from completed calls
        completed_records = session.query(CallRecord).filter_by(status="completed").all()
        avg_score = 0.0
        score_count = 0
        compliance_flag_count = 0
        for rec in completed_records:
            if rec.qa_scores_json:
                try:
                    q = json.loads(rec.qa_scores_json)
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

        # Build dashboard
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

        # LangSmith link
        project = config.langchain_project
        langsmith_url = "https://smith.langchain.com/o/default/projects"
        tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

        langsmith_md = "## LangSmith Integration\n\n"
        if tracing_enabled:
            langsmith_md += (
                f"Tracing is **enabled** for project: "
                f"`{project}`\n\n"
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
                f"LANGCHAIN_PROJECT={project}\n"
                "```\n"
            )

        # Audit log table
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
    finally:
        session.close()


# --- UI ---
def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Call Center Intelligence System",
        theme=gr.themes.Soft(),
    ) as ui:
        gr.Markdown(
            "# Call Center Intelligence System\n\n"
            "AI-powered call center analysis platform. Upload audio recordings to "
            "generate transcripts, summaries, quality scores, and compliance reports."
        )

        # === Tab 1: Analyze Call ===
        with gr.Tab("Analyze Call"):
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(
                        type="numpy",
                        label="Upload or Record Audio",
                        sources=["upload", "microphone"],
                    )
                with gr.Column(scale=1):
                    caller_id = gr.Textbox(
                        label="Caller ID (optional)",
                        placeholder="C-12345",
                    )
                    department = gr.Textbox(
                        label="Department (optional)",
                        placeholder="Billing",
                    )
                    analyze_btn = gr.Button(
                        "Analyze Call",
                        variant="primary",
                        size="lg",
                    )

            gr.Markdown("---")

            status_msg = gr.Markdown(visible=False)

            with gr.Accordion("Full Transcript", open=True):
                transcript_out = gr.Textbox(
                    label="Transcript",
                    lines=15,
                    show_copy_button=True,
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Call Summary")
                    summary_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("## Quality Analysis")
                    qa_out = gr.Markdown()

            with gr.Row():
                json_file = gr.File(label="Download Full Report (JSON)")
                pdf_file = gr.File(label="Download Full Report (PDF)")

            # Show processing message, then run pipeline
            analyze_btn.click(
                fn=lambda: gr.Markdown(
                    value=(
                        "**Processing...** Transcribing audio, "
                        "generating summary and QA scores. "
                        "This takes 3-8 minutes on CPU."
                    ),
                    visible=True,
                ),
                outputs=[status_msg],
            ).then(
                fn=process_call,
                inputs=[audio_input, caller_id, department],
                outputs=[
                    transcript_out,
                    summary_out,
                    qa_out,
                    json_file,
                    pdf_file,
                ],
            ).then(
                fn=lambda: gr.Markdown(visible=False),
                outputs=[status_msg],
            )

        # === Tab 2: Batch Processing ===
        with gr.Tab("Batch Processing"):
            batch_files = gr.Files(label="Upload Multiple Audio Files")
            batch_btn = gr.Button("Process All", variant="primary")
            batch_out = gr.Textbox(label="Results", lines=20)
            batch_btn.click(
                fn=process_batch,
                inputs=[batch_files],
                outputs=[batch_out],
            )

        # === Tab 3: Call History ===
        with gr.Tab("Call History"):
            gr.Markdown("### Processed Calls")
            refresh_btn = gr.Button("Refresh History", variant="secondary")
            history_summary = gr.Markdown("Click refresh to load.")
            history_tbl = gr.Dataframe(
                headers=[
                    "Call ID",
                    "Status",
                    "Date",
                    "Resolution",
                    "Score",
                    "Summary",
                ],
                label="All Processed Calls",
                wrap=True,
                interactive=True,
                column_widths=[
                    "120px",
                    "90px",
                    "140px",
                    "100px",
                    "70px",
                    "300px",
                ],
            )

            refresh_btn.click(
                fn=load_all_history,
                outputs=[history_summary, history_tbl],
            )

            gr.Markdown("---")
            gr.Markdown("### Call Details")
            with gr.Row():
                detail_id = gr.Textbox(
                    label="Enter Call ID (or partial ID)",
                    placeholder="Paste Call ID from table above",
                )
                detail_btn = gr.Button("View Full Details")

            with gr.Accordion("Transcript", open=False):
                det_transcript = gr.Textbox(
                    label="Transcript",
                    lines=12,
                    show_copy_button=True,
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Summary")
                    det_summary = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### Quality Analysis")
                    det_qa = gr.Markdown()

            detail_btn.click(
                fn=get_call_detail,
                inputs=[detail_id],
                outputs=[det_transcript, det_summary, det_qa],
            )

        # === Tab 4: Observability ===
        with gr.Tab("Observability") as obs_tab:
            gr.Markdown(
                "### Pipeline Observability & LangSmith Tracing\n"
                "Monitor pipeline health, audit trail, and "
                "LangSmith integration status."
            )
            obs_refresh = gr.Button("Refresh Dashboard", variant="secondary")

            with gr.Row():
                with gr.Column():
                    obs_metrics = gr.Markdown("Loading...")
                with gr.Column():
                    obs_langsmith = gr.Markdown("Loading...")

            gr.Markdown("### Audit Trail")
            obs_audit = gr.Dataframe(
                headers=[
                    "Timestamp",
                    "Call ID",
                    "Action",
                    "Details",
                ],
                label="Recent Audit Events",
                wrap=True,
                interactive=True,
                column_widths=[
                    "180px",
                    "140px",
                    "120px",
                    "300px",
                ],
            )

            obs_refresh.click(
                fn=get_observability_dashboard,
                outputs=[obs_metrics, obs_langsmith, obs_audit],
            )

            # Auto-load on tab select
            obs_tab.select(
                fn=get_observability_dashboard,
                outputs=[obs_metrics, obs_langsmith, obs_audit],
            )

        # Auto-load on app start
        ui.load(
            fn=get_observability_dashboard,
            outputs=[obs_metrics, obs_langsmith, obs_audit],
        )

    return ui


demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
