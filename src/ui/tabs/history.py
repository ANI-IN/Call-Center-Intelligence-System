"""Tab 3: All MP3 History — browse past analyses (master-detail in one tab)."""

from __future__ import annotations

import logging
import tempfile

import gradio as gr

from src.agents.report import generate_report_json, generate_report_pdf
from src.services.history import get_call_detail, list_calls

logger = logging.getLogger(__name__)

_TABLE_HEADERS = ["Timestamp", "File", "Status", "Overall Score", "Resolution"]
_DETAIL_PLACEHOLDER = "_Select a call from the table above to view its full analysis._"


def _rows_to_display(rows: list[dict]) -> list[list[str]]:
    return [
        [
            r["processed_at"],
            r["audio_filename"],
            r["status"],
            r["overall_score"],
            r["resolution_status"],
        ]
        for r in rows
    ]


def build_history_tab(engine) -> tuple[gr.Dataframe, gr.Markdown]:
    """Build the All MP3 History tab. Returns the table and detail-header
    components so the parent can wire tab-select autoload.
    """
    gr.Markdown(
        "### All MP3 History\n"
        "Every call analyzed in this app is listed below, newest first. "
        "Select a row to load its full transcript, summary, and quality analysis."
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh", variant="secondary")
        gr.Markdown(
            "_Showing up to 200 most recent calls. Older calls remain stored in the database._"
        )

    history_table = gr.Dataframe(
        headers=_TABLE_HEADERS,
        label="Past Calls",
        interactive=False,
        wrap=True,
        row_count=(0, "dynamic"),
        col_count=(len(_TABLE_HEADERS), "fixed"),
    )

    # Hidden state: list of call_ids in the same order as the table rows.
    call_ids_state = gr.State(value=[])

    gr.Markdown("---")

    detail_header = gr.Markdown(_DETAIL_PLACEHOLDER)

    with gr.Accordion("Full Transcript", open=False) as transcript_accordion:
        transcript_out = gr.Textbox(
            label="Transcript",
            lines=15,
            show_copy_button=True,
            interactive=False,
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

    def _load_history():
        rows = list_calls(engine)
        logger.info(f"History tab: loaded {len(rows)} call(s)")
        ids = [r["call_id"] for r in rows]
        return _rows_to_display(rows), ids

    def _on_row_select(call_ids: list[str], evt: gr.SelectData):
        if evt is None or evt.index is None or not call_ids:
            return (
                _DETAIL_PLACEHOLDER,
                gr.update(open=False),
                "",
                "",
                "",
                None,
                None,
            )
        # evt.index can be a [row, col] pair or a single int depending on Gradio version
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if idx < 0 or idx >= len(call_ids):
            logger.warning(f"History tab: row index {idx} out of range for {len(call_ids)} ids")
            return (
                "_Could not load that call._",
                gr.update(open=False),
                "",
                "",
                "",
                None,
                None,
            )
        call_id = call_ids[idx]
        logger.info(f"History tab: loading detail for call_id={call_id}")
        detail = get_call_detail(engine, call_id)
        if detail is None:
            return (
                f"_Call {call_id} not found._",
                gr.update(open=False),
                "",
                "",
                "",
                None,
                None,
            )

        header_md = (
            f"### Call `{detail['call_id']}`\n"
            f"**File:** {detail['audio_filename'] or '—'}  •  "
            f"**Processed at:** {detail['processed_at'] or '—'}  •  "
            f"**Status:** `{detail['status']}`"
        )

        json_path = None
        pdf_path = None
        report = detail.get("report")
        if report is not None:
            try:
                jt = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".json", prefix=f"history_{call_id}_"
                )
                with open(jt.name, "w") as f:
                    f.write(generate_report_json(report))
                json_path = jt.name

                pt = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf", prefix=f"history_{call_id}_"
                )
                with open(pt.name, "wb") as f:
                    f.write(generate_report_pdf(report))
                pdf_path = pt.name
            except Exception:
                logger.exception(f"History tab: failed to regenerate downloads for {call_id}")

        return (
            header_md,
            gr.update(open=True),
            detail["transcript"],
            detail["summary_md"],
            detail["qa_md"],
            json_path,
            pdf_path,
        )

    refresh_btn.click(
        fn=_load_history,
        outputs=[history_table, call_ids_state],
    )

    history_table.select(
        fn=_on_row_select,
        inputs=[call_ids_state],
        outputs=[
            detail_header,
            transcript_accordion,
            transcript_out,
            summary_out,
            qa_out,
            json_file,
            pdf_file,
        ],
    )

    return history_table, call_ids_state


def wire_history_autoload(tab, history_table, call_ids_state, engine):
    """Auto-refresh the table whenever the History tab is selected."""

    def _load():
        rows = list_calls(engine)
        return _rows_to_display(rows), [r["call_id"] for r in rows]

    tab.select(fn=_load, outputs=[history_table, call_ids_state])
