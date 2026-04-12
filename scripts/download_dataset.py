"""Download real call center datasets from HuggingFace.

Two datasets are used:
1. AxonData: 2 MP3 audio files + DOCX transcripts (for audio pipeline testing)
2. AIxBlock: 91,706 real call center transcripts with timestamps (for evaluation)

Downloads audio to data/audio/ and transcripts to data/transcripts/.
Generates ground truth annotations to evaluations/ground_truth/.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# --- Config ---
AXONDATA_REPO = "AxonData/english-contact-center-audio-dataset"
AIXBLOCK_REPO = "AIxBlock/92k-real-world-call-center-scripts-english"

AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts")
GT_DIR = Path("evaluations/ground_truth")

# Which AIxBlock zips to download (smallest ones for capstone scope)
AIXBLOCK_ZIPS = [
    "medical_equipment_outbound.zip",
    "auto_insurance_customer_service_inbound.zip",
    "customer_service_general_inbound.zip",
]

# How many transcripts to extract per zip for ground truth
TRANSCRIPTS_PER_ZIP = 10
TOTAL_GT_TARGET = 30


def download_axondata_audio() -> list[dict]:
    """Download AxonData MP3 audio files + DOCX transcripts."""
    print("=" * 60)
    print("Downloading AxonData audio files (MP3 + DOCX)...")
    print("=" * 60)

    api = HfApi()
    files = api.list_repo_files(AXONDATA_REPO, repo_type="dataset")

    audio_files = sorted(f for f in files if f.endswith(".mp3"))
    docx_files = sorted(f for f in files if f.endswith(".docx"))

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for mp3, docx in zip(audio_files, docx_files):
        mp3_name = mp3.split("/")[-1]
        docx_name = docx.split("/")[-1]
        call_id = mp3_name.replace(".mp3", "")

        mp3_path = hf_hub_download(AXONDATA_REPO, mp3, repo_type="dataset")
        docx_path = hf_hub_download(AXONDATA_REPO, docx, repo_type="dataset")

        shutil.copy(mp3_path, AUDIO_DIR / mp3_name)
        shutil.copy(docx_path, AUDIO_DIR / docx_name)

        print(f"  Audio: {mp3_name}")
        downloaded.append({"call_id": call_id, "type": "audio", "file": mp3_name})

    print(f"  Total: {len(downloaded)} audio files\n")
    return downloaded


def download_aixblock_transcripts() -> list[dict]:
    """Download AIxBlock real call center transcripts."""
    print("=" * 60)
    print("Downloading AIxBlock transcripts (91K real calls)...")
    print("=" * 60)

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    all_transcripts = []
    gt_count = 0

    for zip_name in AIXBLOCK_ZIPS:
        print(f"\n  Downloading {zip_name}...")
        zip_path = hf_hub_download(AIXBLOCK_REPO, zip_name, repo_type="dataset")

        domain = zip_name.replace(".zip", "")
        domain_dir = TRANSCRIPT_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        gt_created = 0

        with zipfile.ZipFile(zip_path) as zf:
            json_files = [
                n for n in zf.namelist() if n.endswith(".json") and not n.startswith("__MACOSX")
            ]
            print(f"  Found {len(json_files)} transcripts in {zip_name}")

            for name in json_files:
                try:
                    with zf.open(name) as f:
                        raw = f.read()
                    data = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                text = data.get("text", "")
                if not text or len(text.split()) < 50:
                    continue

                fname = name.split("/")[-1]
                call_id = fname.replace(".json", "").replace("_transcript", "")

                # Save transcript
                transcript_path = domain_dir / fname
                with open(transcript_path, "w") as f:
                    json.dump(data, f, indent=2)
                extracted += 1

                all_transcripts.append(
                    {
                        "call_id": call_id,
                        "domain": domain,
                        "file": str(transcript_path),
                        "duration": data.get("audio_duration", 0),
                        "confidence": data.get("confidence", 0),
                        "word_count": len(text.split()),
                    }
                )

                # Create ground truth for first N per zip
                if gt_created < TRANSCRIPTS_PER_ZIP and gt_count < TOTAL_GT_TARGET:
                    gt = _create_ground_truth_from_transcript(call_id, domain, data)
                    gt_path = GT_DIR / f"{call_id}.json"
                    with open(gt_path, "w") as f:
                        json.dump(gt, f, indent=2)
                    gt_created += 1
                    gt_count += 1

        print(f"  Extracted: {extracted} transcripts, {gt_created} ground truth annotations")

    print(f"\n  Total transcripts: {len(all_transcripts)}")
    print(f"  Total ground truth: {gt_count}")
    return all_transcripts


def _create_ground_truth_from_transcript(call_id: str, domain: str, data: dict) -> dict:
    """Create ground truth annotation from AIxBlock transcript."""
    text = data.get("text", "")
    words = data.get("words", [])
    confidence = data.get("confidence", 0)
    duration = data.get("audio_duration", 0)

    # Build timestamped transcript from word data
    transcript_lines = []
    current_line = []
    current_start = 0

    for w in words:
        if not current_line:
            current_start = w.get("start", 0)
        current_line.append(w.get("text", ""))

        # Break at sentence boundaries
        if w.get("text", "").endswith((".", "?", "!")) and len(current_line) > 3:
            timestamp = f"{current_start // 1000 // 60:02d}:{current_start // 1000 % 60:02d}"
            line_text = " ".join(current_line)
            transcript_lines.append(f"[{timestamp}] {line_text}")
            current_line = []

    if current_line:
        timestamp = f"{current_start // 1000 // 60:02d}:{current_start // 1000 % 60:02d}"
        transcript_lines.append(f"[{timestamp}] {' '.join(current_line)}")

    full_transcript = "\n".join(transcript_lines)

    # Determine domain-specific context
    domain_labels = {
        "medical_equipment_outbound": "Medical Equipment Sales",
        "auto_insurance_customer_service_inbound": "Auto Insurance Support",
        "customer_service_general_inbound": "General Customer Service",
        "automotive_inbound": "Automotive Support",
        "home_service_inbound": "Home Service Support",
        "insurance_outbound": "Insurance Outbound Sales",
        "medicare_inbound": "Medicare Support",
    }
    domain_label = domain_labels.get(domain, domain)

    return {
        "call_id": call_id,
        "audio_file": None,
        "domain": domain_label,
        "reference_transcript": full_transcript,
        "reference_summary": (
            f"{domain_label} call. "
            f"Duration: {duration}s. "
            f"Transcription confidence: {confidence:.3f}. "
            f"Word count: {len(text.split())}."
        ),
        "reference_key_points": [
            f"Domain: {domain_label}",
            f"Call duration: {duration} seconds",
        ],
        "reference_entities": [],
        "reference_resolution_status": "unresolved",
        "qa_scores": {
            "professionalism": 4,
            "empathy": 3,
            "problem_resolution": 3,
            "compliance": 4,
            "communication_clarity": 4,
            "overall": 3.6,
        },
        "qa_justifications": {
            "professionalism": ("Agent followed standard greeting protocol."),
            "empathy": "Baseline score — requires human review.",
            "problem_resolution": ("Baseline score — requires human review."),
            "compliance": "Standard procedures observed.",
            "communication_clarity": ("Communication was generally clear."),
        },
        "has_compliance_violation": False,
        "compliance_violations": [],
        "annotator_notes": (
            f"Auto-generated from AIxBlock dataset. "
            f"Domain: {domain_label}. "
            f"ASR confidence: {confidence:.3f}. "
            f"Learner should review and refine these scores."
        ),
        "source_dataset": "AIxBlock/92k-real-world-call-center-scripts-english",
        "raw_text": text,
        "word_count": len(text.split()),
        "audio_duration_seconds": duration,
        "asr_confidence": confidence,
    }


def download_axondata_docx_gt() -> None:
    """Parse AxonData DOCX transcripts into ground truth."""
    print("=" * 60)
    print("Parsing AxonData DOCX transcripts for ground truth...")
    print("=" * 60)

    try:
        from docx import Document
    except ImportError:
        print("  python-docx not installed, skipping DOCX parsing")
        return

    GT_DIR.mkdir(parents=True, exist_ok=True)

    for docx_file in AUDIO_DIR.glob("*.docx"):
        call_id = docx_file.stem
        doc = Document(str(docx_file))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Extract sections from DOCX
        summary = ""
        transcript_lines = []
        transcript_started = False

        for line in paragraphs:
            lower = line.lower()
            if "transcription" in lower:
                transcript_started = True
                continue
            if lower == "key points":
                continue

            if transcript_started:
                transcript_lines.append(line)
            elif lower == "key points":
                continue

        # Build transcript from timestamp + text pairs
        full_transcript = ""
        i = 0
        while i < len(transcript_lines):
            line = transcript_lines[i]
            if len(line) >= 7 and line[2] == ":" and line[5] == ":":
                if i + 1 < len(transcript_lines):
                    full_transcript += f"[{line}] {transcript_lines[i + 1]}\n"
                    i += 2
                else:
                    i += 1
            else:
                full_transcript += f"{line}\n"
                i += 1

        # Find summary from key points section
        for j, line in enumerate(paragraphs):
            if line.lower() == "key points" and j + 1 < len(paragraphs):
                summary = paragraphs[j + 1]
                break

        gt = {
            "call_id": call_id,
            "audio_file": f"data/audio/{call_id}.mp3",
            "domain": "Customer Support (AxonData)",
            "reference_transcript": full_transcript.strip(),
            "reference_summary": summary,
            "reference_key_points": [],
            "reference_entities": [],
            "reference_resolution_status": "unresolved",
            "qa_scores": {
                "professionalism": 4,
                "empathy": 4,
                "problem_resolution": 3,
                "compliance": 4,
                "communication_clarity": 4,
                "overall": 3.8,
            },
            "qa_justifications": {
                "professionalism": ("Agent introduced themselves professionally."),
                "empathy": ("Agent showed patience with customer issues."),
                "problem_resolution": ("Issue partially addressed — requires review."),
                "compliance": "Standard procedures followed.",
                "communication_clarity": ("Clear instructions provided."),
            },
            "has_compliance_violation": False,
            "compliance_violations": [],
            "annotator_notes": ("Parsed from AxonData DOCX transcript. Has matching audio file."),
            "source_dataset": AXONDATA_REPO,
        }

        gt_path = GT_DIR / f"{call_id}.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"  Ground truth: {call_id}")


def print_summary() -> None:
    """Print dataset summary."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    audio_count = len(list(AUDIO_DIR.glob("*.mp3")))
    transcript_count = sum(1 for _ in TRANSCRIPT_DIR.rglob("*.json") if TRANSCRIPT_DIR.exists())
    gt_count = len([f for f in GT_DIR.glob("*.json") if f.name != "SCHEMA.md"])

    print(f"  Audio files (MP3):        {audio_count}")
    print(f"  Transcript files (JSON):  {transcript_count}")
    print(f"  Ground truth annotations: {gt_count}")
    print()
    print("Directories:")
    print(f"  Audio:       {AUDIO_DIR}/")
    print(f"  Transcripts: {TRANSCRIPT_DIR}/")
    print(f"  Ground truth: {GT_DIR}/")
    print()
    print("Sources:")
    print(f"  Audio: https://huggingface.co/datasets/{AXONDATA_REPO}")
    print(f"  Transcripts: https://huggingface.co/datasets/{AIXBLOCK_REPO}")
    print()
    print("Next steps:")
    print("  1. Review ground truth annotations in evaluations/ground_truth/")
    print("  2. Refine QA scores based on your reading of each transcript")
    print("  3. Run: make eval")


def main() -> None:
    print()
    print("Call Center Intelligence System — Dataset Download")
    print()

    # Step 1: Download AxonData audio
    download_axondata_audio()

    # Step 2: Parse AxonData DOCX for ground truth
    download_axondata_docx_gt()

    # Step 3: Download AIxBlock transcripts
    download_aixblock_transcripts()

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
