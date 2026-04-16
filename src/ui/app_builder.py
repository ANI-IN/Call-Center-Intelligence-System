"""Gradio app builder — assembles all tabs into the main UI."""

from __future__ import annotations

import gradio as gr

from src.services.observability import get_observability_dashboard
from src.ui.tabs.analyze import build_analyze_tab
from src.ui.tabs.history import build_history_tab, wire_history_autoload
from src.ui.tabs.observability import (
    build_observability_tab,
    wire_observability_autoload,
)


def build_app(config, workflow, engine, audit) -> gr.Blocks:
    """Build the complete Gradio application."""
    with gr.Blocks(
        title="Call Center Intelligence System",
        theme=gr.themes.Soft(),
    ) as ui:
        gr.Markdown(
            "# Call Center Intelligence System\n\n"
            "AI-powered call center analysis platform. Upload audio recordings to "
            "generate transcripts, summaries, quality scores, and compliance reports."
        )

        with gr.Tab("Analyze Call"):
            build_analyze_tab(workflow, engine, audit)

        with gr.Tab("All MP3 History") as history_tab:
            history_table, call_ids_state = build_history_tab(engine)
            wire_history_autoload(history_tab, history_table, call_ids_state, engine)

        with gr.Tab("Observability") as obs_tab:
            obs_metrics, obs_langsmith, obs_audit = build_observability_tab(
                engine, config.langchain_project
            )
            wire_observability_autoload(
                obs_tab,
                obs_metrics,
                obs_langsmith,
                obs_audit,
                engine,
                config.langchain_project,
            )

        ui.load(
            fn=lambda: get_observability_dashboard(engine, config.langchain_project),
            outputs=[obs_metrics, obs_langsmith, obs_audit],
        )

    return ui
