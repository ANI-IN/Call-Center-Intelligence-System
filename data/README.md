# Dataset Setup

This project uses **two real call center datasets** — no synthetic data.

## Automatic Download

```bash
make download-data
# OR
python scripts/download_dataset.py
```

This downloads everything and generates 32 ground truth annotations automatically.

## Dataset 1: AxonData Audio + Transcripts (Audio Pipeline)

- **Source:** https://huggingface.co/datasets/AxonData/english-contact-center-audio-dataset
- **What:** 2 real call center MP3 recordings with DOCX transcripts
- **Use:** Testing the full audio-to-insight pipeline (Whisper STT + diarization)
- **License:** CC BY-NC 4.0

| Call ID | Domain | Duration |
|---|---|---|
| 1735404531.458927 | Customer Support (Finance) | ~11.5 min |
| 1755884171.51632 | Billing Support (Crypto/Tax) | ~14 min |

## Dataset 2: AIxBlock 92K Transcripts (Evaluation at Scale)

- **Source:** https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english
- **What:** 91,706 real call center transcripts with word-level timestamps and confidence scores
- **Use:** Large-scale evaluation of summarization and QA scoring agents
- **License:** CC BY-NC 4.0
- **PII:** Already redacted by the dataset provider

Downloaded domains (3,700+ transcripts):

| Domain | Transcripts | Avg Duration |
|---|---|---|
| Medical Equipment Outbound | ~738 | ~500s |
| Auto Insurance Inbound | ~1,746 | ~300-600s |
| Customer Service General | ~1,217 | ~200-500s |

## Ground Truth

32 ground truth annotations auto-generated in `evaluations/ground_truth/`:
- 2 from AxonData (with matching audio files)
- 30 from AIxBlock (transcript-only, 10 per domain)

Learners should **review and refine** these baseline annotations for accurate evaluation.
