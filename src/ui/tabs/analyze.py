"""Tab 1: Single call analysis UI."""

from __future__ import annotations

import logging

import gradio as gr

from src.services.pipeline import process_call

logger = logging.getLogger(__name__)


def build_analyze_tab(workflow, engine, audit) -> None:
    """Build the Analyze Call tab inside an existing gr.Tab context."""
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                # filepath: avoids decoding/re-encoding to WAV which can blow a
                # small MP3 past the 50MB intake limit.
                type="filepath",
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

    def _show_processing():
        return gr.update(
            value=(
                "**Processing your call...** This will take approximately "
                "**5-10 minutes** depending on the audio length. "
                "The system is transcribing the audio, generating a summary, "
                "and scoring the call quality. Please be patient and do not "
                "refresh the page."
            ),
            visible=True,
        )

    def _run_pipeline(audio, cid, dept):
        logger.info(f"Analyze tab: running pipeline (caller={cid or '-'}, dept={dept or '-'})")
        result = process_call(audio, cid, dept, workflow, engine, audit)
        if not result.success:
            error_text = result.error or "Unknown error"
            logger.warning(f"Analyze tab: pipeline returned failure — {error_text}")
            gr.Warning(error_text)
            # Show error in the visible status banner; clear analysis fields so
            # users do not see stale content from a previous run.
            error_md = gr.update(
                value=f"### Pipeline failed\n\n```\n{error_text}\n```",
                visible=True,
            )
            return error_md, "", "", "", None, None
        logger.info("Analyze tab: pipeline succeeded — rendering results")
        success_md = gr.update(
            value="**Analysis complete.** See transcript, summary, QA, and downloads below.",
            visible=True,
        )
        return (
            success_md,
            result.transcript,
            result.summary_md,
            result.qa_md,
            result.json_path,
            result.pdf_path,
        )

    analyze_btn.click(
        fn=_show_processing,
        outputs=[status_msg],
    ).then(
        fn=_run_pipeline,
        inputs=[audio_input, caller_id, department],
        outputs=[status_msg, transcript_out, summary_out, qa_out, json_file, pdf_file],
    )
