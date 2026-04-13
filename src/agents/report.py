from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.database.connection import get_engine, get_session, init_db
from src.database.models import CallRecord
from src.graph.state import (
    CallReport,
    IntakeResult,
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
)


def compile_report(
    intake: IntakeResult,
    transcription: TranscriptionResult,
    summary: SummaryResult,
    qa_scores: QAScoreResult,
    trace_id: str,
) -> CallReport:
    return CallReport(
        call_id=intake.call_id,
        intake=intake,
        transcription=transcription,
        summary=summary,
        qa_scores=qa_scores,
        processed_at=datetime.now(),
        trace_id=trace_id,
        status="completed",
    )


def persist_report(
    report: CallReport,
    db_path: str | None = None,
    encryption_key: str | None = None,
    engine=None,
) -> None:
    if engine is None:
        engine = get_engine(db_path, encryption_key)
        init_db(engine)
    session = get_session(engine)
    try:
        record = CallRecord(
            call_id=str(report.call_id),
            status=report.status,
            audio_filename=Path(report.intake.audio_path).name if report.intake.audio_path else "",
            transcript_text=report.transcription.full_text,
            summary_json=report.summary.model_dump_json(),
            qa_scores_json=report.qa_scores.model_dump_json(),
            report_json=report.model_dump_json(),
            processed_at=report.processed_at,
            trace_id=report.trace_id,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def generate_report_json(report: CallReport) -> str:
    return report.model_dump_json(indent=2)


def generate_report_pdf(report: CallReport) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story: list = []

    story.append(Paragraph(f"Call Report: {report.call_id}", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(f"Purpose: {report.summary.call_purpose}", styles["Normal"]))
    story.append(
        Paragraph(f"Resolution: {report.summary.resolution_status.value}", styles["Normal"])
    )
    story.append(Paragraph(f"Sentiment: {report.summary.sentiment_trajectory}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("QA Scores", styles["Heading2"]))
    scores = report.qa_scores
    for dim in [
        "professionalism",
        "empathy",
        "problem_resolution",
        "compliance",
        "communication_clarity",
    ]:
        dim_score = getattr(scores, dim)
        story.append(
            Paragraph(f"{dim}: {dim_score.score}/5 - {dim_score.justification}", styles["Normal"])
        )
    story.append(Paragraph(f"Overall: {scores.overall_score}/5", styles["Normal"]))
    story.append(Spacer(1, 12))

    if scores.compliance_flags:
        story.append(Paragraph("Compliance Flags", styles["Heading2"]))
        for flag in scores.compliance_flags:
            story.append(
                Paragraph(
                    f"[{flag.severity.upper()}] {flag.violation} "
                    f"(ref: {flag.transcript_reference})",
                    styles["Normal"],
                )
            )

    doc.build(story)
    return buf.getvalue()
