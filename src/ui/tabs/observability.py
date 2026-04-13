"""Tab 4: Observability dashboard UI."""

from __future__ import annotations

import gradio as gr

from src.services.observability import get_observability_dashboard


def build_observability_tab(
    engine, langchain_project: str
) -> tuple[gr.Markdown, gr.Markdown, gr.Dataframe]:
    """Build the Observability tab. Returns components for auto-load wiring."""
    gr.Markdown(
        "### Pipeline Observability & LangSmith Tracing\n"
        "Monitor pipeline health, audit trail, and LangSmith integration status."
    )
    obs_refresh = gr.Button("Refresh Dashboard", variant="secondary")

    with gr.Row():
        with gr.Column():
            obs_metrics = gr.Markdown("Loading...")
        with gr.Column():
            obs_langsmith = gr.Markdown("Loading...")

    gr.Markdown("### Audit Trail")
    obs_audit = gr.Dataframe(
        headers=["Timestamp", "Call ID", "Action", "Details"],
        label="Recent Audit Events",
        wrap=True,
        interactive=False,
    )

    def _refresh():
        return get_observability_dashboard(engine, langchain_project)

    obs_refresh.click(fn=_refresh, outputs=[obs_metrics, obs_langsmith, obs_audit])

    return obs_metrics, obs_langsmith, obs_audit


def wire_observability_autoload(
    tab, obs_metrics, obs_langsmith, obs_audit, engine, langchain_project
):
    """Wire tab-select auto-load for observability."""
    tab.select(
        fn=lambda: get_observability_dashboard(engine, langchain_project),
        outputs=[obs_metrics, obs_langsmith, obs_audit],
    )
