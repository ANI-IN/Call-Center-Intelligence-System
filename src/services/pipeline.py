"""Pipeline orchestration service — no UI dependencies."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.agents.report import generate_report_json, generate_report_pdf, persist_report
from src.graph.state import AudioInput, CallReport
from src.security.audit import AuditLogger
from src.utils.formatters import format_qa, format_summary

logger = logging.getLogger(__name__)

_temp_files: list[str] = []
_MAX_TEMP_FILES = 50


def _cleanup_old_temp_files() -> None:
    """Remove old temp files to prevent disk buildup."""
    global _temp_files
    if len(_temp_files) > _MAX_TEMP_FILES:
        to_remove = _temp_files[:-_MAX_TEMP_FILES]
        _temp_files = _temp_files[-_MAX_TEMP_FILES:]
        for f in to_remove:
            try:
                os.unlink(f)
            except OSError:
                pass


@dataclass
class PipelineResult:
    """Result of processing a single call."""

    success: bool
    transcript: str = ""
    summary_md: str = ""
    qa_md: str = ""
    json_path: str | None = None
    pdf_path: str | None = None
    error: str | None = None


def process_call(
    audio_file,
    caller_id: str | None,
    department: str | None,
    workflow,
    engine,
    audit: AuditLogger,
) -> PipelineResult:
    """Process audio through the full pipeline. Returns structured result."""
    if audio_file is None:
        logger.info("process_call: no audio provided")
        return PipelineResult(success=False, error="No file uploaded.")

    _cleanup_old_temp_files()
    logger.info(f"process_call: received audio of type {type(audio_file).__name__}")

    # Gradio type="numpy" returns (sample_rate, numpy_array). We re-encode to a
    # temp WAV so downstream agents see a real file. Note: this can balloon a
    # small MP3 into a much larger WAV, so the UI now uses type="filepath" for
    # uploads and we only hit this branch for microphone recordings.
    if isinstance(audio_file, tuple):
        import numpy as np
        import soundfile as sf

        sr, arr = audio_file
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            logger.info("process_call: empty microphone recording")
            return PipelineResult(success=False, error="Empty audio recording.")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, arr, sr)
        _temp_files.append(tmp.name)
        audio_file = tmp.name
        logger.info(
            f"process_call: wrote microphone audio to {tmp.name} "
            f"({arr.size} samples @ {sr}Hz)"
        )

    audio_path = Path(str(audio_file))
    if not audio_path.exists():
        logger.warning(f"process_call: file not found at {audio_file}")
        return PipelineResult(success=False, error=f"File not found: {audio_file}")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    logger.info(
        f"process_call: invoking workflow with {audio_path.name} "
        f"({len(audio_data)} bytes, caller={caller_id or '-'}, dept={department or '-'})"
    )

    audio_input = AudioInput(
        audio_data=audio_data,
        filename=audio_path.name,
        caller_id=caller_id or None,
        department=department or None,
        timestamp=datetime.now(),
    )

    try:
        result = workflow.invoke({"audio_input": audio_input})
    except Exception as e:
        logger.exception("process_call: workflow.invoke raised an exception")
        return PipelineResult(success=False, error=f"Pipeline error: {e}")

    status = result.get("status", "unknown")
    logger.info(f"process_call: workflow returned status={status}")

    if status == "failed":
        error = result.get("error") or "Unknown error"
        logger.warning(f"process_call: pipeline failed — {error}")
        audit.log(call_id="unknown", action="pipeline_failed", user="app", details={"error": error})
        return PipelineResult(success=False, error=f"Pipeline failed: {error}")

    report: CallReport | None = result.get("report")
    if report is None:
        logger.warning("process_call: status was not 'failed' but no report was produced")
        return PipelineResult(success=False, error="No report generated.")

    try:
        persist_report(report, engine=engine)
    except Exception:
        logger.exception(
            "process_call: persist_report failed (continuing to return result to user)"
        )

    audit.log(
        call_id=str(report.call_id),
        action="completed",
        user="app",
        details={"status": status},
    )
    logger.info(f"process_call: completed call_id={report.call_id}")

    # Format transcript
    lines = []
    for seg in report.transcription.segments:
        tag = " [LOW CONF]" if seg.low_confidence else ""
        m = int(seg.start_time // 60)
        s = int(seg.start_time % 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg.speaker}: {seg.text}{tag}")
    transcript = "\n".join(lines)

    summary_md = format_summary(report.summary.model_dump())
    qa_md = format_qa(report.qa_scores.model_dump())

    # Generate downloadable files
    json_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="report_")
    with open(json_tmp.name, "w") as f:
        f.write(generate_report_json(report))
    _temp_files.append(json_tmp.name)

    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="report_")
    with open(pdf_tmp.name, "wb") as f:
        f.write(generate_report_pdf(report))
    _temp_files.append(pdf_tmp.name)

    intake_path = report.intake.audio_path
    if intake_path and os.path.exists(intake_path) and "/tmp/" in intake_path:
        _temp_files.append(intake_path)

    return PipelineResult(
        success=True,
        transcript=transcript,
        summary_md=summary_md,
        qa_md=qa_md,
        json_path=json_tmp.name,
        pdf_path=pdf_tmp.name,
    )
