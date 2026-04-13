"""Call Center Intelligence System — application entrypoint."""

# ruff: noqa: E402
from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from src.agents.transcription import _get_whisper_model
from src.database.connection import get_engine, init_db
from src.graph.workflow import compile_workflow
from src.security.audit import AuditLogger
from src.ui.app_builder import build_app
from src.utils.config import load_config

# --- Startup: load everything ONCE ---
config = load_config()

_engine = get_engine(str(config.db_path), config.db_encryption_key)
init_db(_engine)

print(f"Loading Whisper model ({config.whisper_model_size})...")
_get_whisper_model(config.whisper_model_size)
print("Whisper model loaded.")

_workflow = compile_workflow(config, db_engine=_engine)
_audit = AuditLogger(engine=_engine)
print("Pipeline ready.")

demo = build_app(config, _workflow, _engine, _audit)

if __name__ == "__main__":
    host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    demo.launch(server_name=host, server_port=7860, ssr_mode=False)
