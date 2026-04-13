"""Shared display formatting for summaries, QA scores, and timestamps."""

from __future__ import annotations


def secs_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_summary(d: dict) -> str:
    """Format a summary dict as markdown for display."""
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


def format_qa(d: dict) -> str:
    """Format QA scores dict as markdown for display."""
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

    lines.append("---")
    lines.append("### Compliance Flags")
    flags = d.get("compliance_flags", [])
    if flags:
        for f in flags:
            sev = f.get("severity", "").upper()
            viol = f.get("violation", "")
            ref = f.get("transcript_reference", "")
            icon = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🔶", "CRITICAL": "🔴"}.get(sev, "⚠️")
            lines.append(f"- {icon} **[{sev}]** {viol} *(ref: {ref})*")
    else:
        lines.append("No compliance issues detected.")

    return "\n".join(lines)
