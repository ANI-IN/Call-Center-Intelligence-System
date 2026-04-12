# Ground Truth Annotation Schema

Each annotated call is a JSON file named `{call_id}.json` with the following structure.
This must match the Pydantic models used by the pipeline.

## Required Fields

```json
{
  "call_id": "uuid-string",
  "audio_file": "relative/path/to/audio.wav",
  "reference_transcript": "Full corrected transcript text...",
  "reference_summary": "The customer called to...",
  "reference_key_points": ["Point 1", "Point 2"],
  "reference_entities": ["$45.99", "April billing cycle"],
  "reference_resolution_status": "resolved",
  "qa_scores": {
    "professionalism": 4,
    "empathy": 5,
    "problem_resolution": 4,
    "compliance": 3,
    "communication_clarity": 4,
    "overall": 4.0
  },
  "qa_justifications": {
    "professionalism": "Agent used proper greeting...",
    "empathy": "Agent acknowledged frustration at 1:23...",
    "problem_resolution": "Root cause identified...",
    "compliance": "Hold procedure not followed at 2:30...",
    "communication_clarity": "Clear explanations..."
  },
  "has_compliance_violation": true,
  "compliance_violations": [
    {
      "violation": "Hold procedure not followed",
      "severity": "medium",
      "transcript_reference": "2:30-2:45"
    }
  ],
  "annotator_notes": "Optional free text notes about this call"
}
```

## Instructions for Annotators

1. Listen to the audio file completely before annotating
2. Write the reference transcript by correcting Whisper output -- note errors
3. Write the reference summary following the exact schema
4. Score each QA dimension 1-5 with justification citing timestamps
5. Flag any compliance violations with severity and transcript reference
6. Minimum 25 calls must be annotated
