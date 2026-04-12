from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_previous"),
    (r"ignore\s+(all\s+)?prior\s+instructions", "ignore_prior"),
    (r"disregard\s+(all\s+)?prior\s+instructions", "disregard_prior"),
    (r"forget\s+(everything|all)\s+(above|previous)", "forget_previous"),
    (r"(print|reveal|show|output|display)\s+(your\s+)?(system\s+)?prompt", "prompt_leak"),
    (r"(what\s+is|what's)\s+your\s+system\s+prompt", "prompt_leak_question"),
    (r"system\s*prompt\s*[?:]", "system_prompt_inject"),
    (r"<<\s*SYS\s*>>", "llama_system_tag"),
    (r"\[INST\]", "llama_inst_tag"),
    (r"\[/INST\]", "llama_inst_close_tag"),
    (r"you\s+are\s+(now|no\s+longer)", "role_switch"),
    (r"new\s+instructions?\s*:", "new_instructions"),
    (r"(DAN|do\s+anything\s+now)\s+(mode\s+)?enabled", "dan_mode"),
    (r"jailbreak", "jailbreak"),
    (r"override\s+safety", "override_safety"),
    (r"ignore\s+(the\s+)?(call\s+)?transcript", "ignore_transcript"),
    (r"\\n\\nHuman:.*\\n\\nAssistant:", "conversation_inject"),
    (r"act\s+as\s+my\s+deceased", "social_engineering"),
    (r"translate\s+the\s+above\s+instructions", "translate_attack"),
    (r"ignore\s+safety\s+guidelines", "ignore_safety"),
    (r"SYSTEM:\s*Override", "system_override"),
    (r"reveal\s+(all\s+)?(secrets|data|instructions)", "reveal_attack"),
]


@dataclass
class InjectionDetectionResult:
    injection_detected: bool
    matched_patterns: list[str] = field(default_factory=list)
    flagged_text: str = ""


def detect_injection(text: str) -> InjectionDetectionResult:
    matched: list[str] = []
    for pattern, name in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(name)

    return InjectionDetectionResult(
        injection_detected=len(matched) > 0,
        matched_patterns=matched,
        flagged_text=text if matched else "",
    )
