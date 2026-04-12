# Production Call Center Intelligence System — Capstone Design Spec

## 1. Project Overview & Business Context

### Project Title
**Production Call Center Intelligence System**

### What Learners Build
An end-to-end, production-grade system that ingests real call center audio recordings, transcribes them, extracts structured insights via a multi-agent LangGraph pipeline, evaluates agent accuracy against human ground truth and LLM judges, secures all data handling for PII compliance, and deploys as a publicly accessible Gradio application with full CI/CD and observability.

### Target Audience
Mixed — junior developers, mid-level engineers transitioning to AI/ML, and experienced ML engineers learning agentic patterns. The project is intensive and milestone-gated so all levels are challenged appropriately.

### Prerequisites
Learners must already have working knowledge of: LangGraph, LangChain, LangSmith, HuggingFace ecosystem.

### Business Context
- A call center processes ~5,000 calls/day. Manual QA covers <5% of calls, takes ~15 min per call, and suffers from inter-rater inconsistency (40-60% agreement on quality scores).
- This system targets: 100% call coverage, <2 min processing per call, >85% agreement with human QA scores, full audit trail for compliance.
- The output is not a demo — it is a system a QA team lead could actually point at a day's worth of calls and get actionable, trustworthy results.

### What Makes This Production-Grade (Not a Toy Project)
- Real audio data, not synthetic
- Accuracy measured against human annotations, not vibes
- Security posture as if handling real PII
- Deployed and accessible, not a localhost notebook
- Observable via LangSmith, not a black box
- CI/CD pipeline, not manual deploys

### Timeline
Milestone-based, no fixed timeline. Learners progress at their own pace but must meet gate criteria at each milestone before advancing.

---

## 2. Technical Architecture

### Mandatory Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** | Multi-agent state machine, conditional routing, retries |
| LLM Framework | **LangChain** | Prompt management, output parsing, chain composition |
| Observability | **LangSmith** | Trace logging, latency tracking, cost monitoring, eval runs |
| Model Hub | **HuggingFace** | Whisper models for STT, embedding models for similarity evals |
| LLM Provider | **OpenAI GPT-4o** | Primary LLM (learners may add Claude/Gemini as fallback) |
| UI | **Gradio** | Upload audio, view results, download reports |
| Deployment | **HuggingFace Spaces** | Public hosting for Gradio app |
| CI/CD | **GitHub Actions** | Automated tests, linting, deployment pipeline |

### Architecture Pattern — LangGraph State Machine

```
Audio Upload → [Intake Node] → [Transcription Node] → [Summarization Node] → [QA Scoring Node] → [Report Node]
                    │                   │                      │                     │
                    v                   v                      v                     v
              Validation           Whisper STT          Structured Summary     Rubric Scores
              PII Detection        Speaker Diarization  Key Topics/Entities    Tone/Empathy/Resolution
              Format Check         Confidence Scores    Action Items           Compliance Flags
                    │                   │                      │                     │
                    └───────────────────┴──────────────────────┴─────────────────────┘
                                                    │
                                          LangSmith Tracing (every node)
                                                    │
                                              Gradio Dashboard
```

### Key Architectural Decisions

1. **LangGraph over CrewAI** — state machine gives explicit control over routing, retries, and conditional logic. No magic orchestration.
2. **Each node is independently testable** — takes typed input, returns typed output via Pydantic models. No hidden state leakage between nodes.
3. **LangSmith tracing on every node** — not optional. Every LLM call, every input/output, every latency measurement is logged.
4. **Fallback strategy** — if primary LLM fails, route to fallback model. If transcription confidence is below threshold, flag for human review rather than passing garbage downstream.
5. **No in-memory state** — all call data persisted to SQLite (local) or PostgreSQL (deployed). System can restart without losing processed calls.

### Data Flow — Typed Contracts Between Nodes

```
AudioInput (bytes, metadata)
    → IntakeResult (validated_metadata, pii_scan_result, format_ok)
        → TranscriptionResult (text, segments[], speaker_labels[], confidence)
            → SummaryResult (summary, key_topics[], action_items[], entities[])
                → QAScoreResult (tone_score, empathy_score, resolution_score, compliance_flags[], overall_score)
                    → CallReport (all above combined, timestamp, trace_id)
```

Every one of these is a **Pydantic model with validation**. No raw dicts flowing between agents.

---

## 3. Agent Design (LangGraph Nodes)

Each agent is a LangGraph node with a single responsibility, typed input/output, error handling, and LangSmith tracing.

### Node 1: Intake Agent

- **Input:** Raw audio file + optional metadata (caller ID, timestamp, department)
- **Responsibility:**
  - Validate audio format (WAV, MP3, FLAC, M4A) and reject unsupported formats with clear error
  - Validate file size (max 50MB) and duration (max 60 min)
  - Run PII pre-scan on any attached metadata (redact before processing)
  - Extract audio properties: duration, sample rate, channel count
  - Generate a unique `call_id` (UUID) that follows the call through every downstream node
- **Output:** `IntakeResult` Pydantic model
- **Failure mode:** Returns validation error with specific reason — does NOT pass invalid data downstream

### Node 2: Transcription Agent

- **Input:** `IntakeResult` with validated audio reference
- **Responsibility:**
  - Transcribe audio using HuggingFace Whisper (locally hosted, not API — learners must handle model loading and inference)
  - Produce word-level timestamps and segment-level confidence scores
  - Perform speaker diarization (minimum: distinguish Agent vs Customer using a HuggingFace diarization model)
  - If segment confidence < configurable threshold (default 0.6), flag that segment as `low_confidence`
  - Calculate overall transcription confidence score
- **Output:** `TranscriptionResult` with full transcript, segments with timestamps, speaker labels, confidence scores
- **Failure mode:** If >40% of segments are `low_confidence`, flag entire call for human review and halt pipeline for that call

### Node 3: Summarization Agent

- **Input:** `TranscriptionResult`
- **Responsibility:**
  - Generate a structured summary (not free-form — must follow a defined schema):
    - **Call Purpose:** 1-2 sentences on why the customer called
    - **Key Discussion Points:** Bullet list of topics covered
    - **Action Items:** Specific next steps with ownership (agent/customer/system)
    - **Resolution Status:** Resolved / Unresolved / Escalated
    - **Customer Sentiment Trajectory:** How sentiment changed during the call (e.g., Frustrated → Satisfied)
  - Extract named entities: product names, account references, dates, amounts
  - All outputs via structured output / function calling — no regex parsing of free text
- **Output:** `SummaryResult` Pydantic model
- **Failure mode:** If LLM returns malformed output, retry up to 2 times with stricter prompt. After 3 failures, log error to LangSmith and mark call as `summary_failed`

### Node 4: QA Scoring Agent

- **Input:** `TranscriptionResult` + `SummaryResult`
- **Responsibility:**
  - Score the call agent's performance on a defined rubric (each scored 1-5 with justification):
    - **Professionalism:** Appropriate language, no interruptions, proper greeting/closing
    - **Empathy:** Acknowledged customer's feelings, active listening indicators
    - **Problem Resolution:** Identified root cause, provided solution, confirmed understanding
    - **Compliance:** Followed required disclosures, verification steps, hold procedures
    - **Communication Clarity:** Clear explanations, avoided jargon, confirmed understanding
  - Each score MUST include a text justification citing specific transcript segments
  - Produce an overall weighted score (learners define and justify their weighting)
  - Flag critical compliance violations separately (these are not just low scores — they are alerts)
- **Output:** `QAScoreResult` Pydantic model with individual scores, justifications, overall score, compliance flags
- **Failure mode:** Same retry logic as summarization. Scores outside 1-5 range are validation errors caught by Pydantic.

### Node 5: Router / Conditional Logic (LangGraph Edges)

Not a separate agent — this is LangGraph's conditional edge logic:
- After Intake: route to error output if validation fails
- After Transcription: route to human review queue if confidence too low
- After Summarization: retry or fail based on output validation
- After QA Scoring: flag for supervisor review if compliance violations detected
- **Fallback LLM routing:** If primary model (GPT-4o) fails or times out (30s), route to fallback model. Learners must implement at least one fallback.

### Node 6: Report Generation Agent

- **Input:** All upstream results
- **Responsibility:**
  - Compile final `CallReport` combining all results
  - Persist to database (SQLite locally, PostgreSQL in production)
  - Generate a downloadable PDF/JSON report
  - Push trace summary to LangSmith
- **Output:** `CallReport` + persisted record + downloadable artifact

---

## 4. Evaluation & Accuracy (Top Priority)

This is the hardest and most important section. Learners must prove their system works, not just demo it.

### 4.1 Ground Truth Annotation (Human Baseline)

Learners must create a gold-standard evaluation set:

- Select **25-30 calls** from the curated dataset
- For each call, manually create:
  - **Reference summary** following the exact schema the Summarization Agent outputs
  - **Reference QA scores** (1-5 on each rubric dimension with justification)
  - **Reference transcript** (corrected version of Whisper output, noting errors)
- Annotation must be done by the learner — this forces them to deeply understand what "good output" looks like
- Store annotations in a structured format (`evaluations/ground_truth/call_id.json`) matching Pydantic model schemas

**Why this matters:** Without ground truth, "accuracy" is meaningless. Learners who skip this are building demo-ware, not production systems.

### 4.2 Automated Metrics (Deterministic)

**Transcription Accuracy:**
- **Word Error Rate (WER)** against reference transcripts — target: <15%
- **Speaker Diarization Error Rate (DER)** — target: <20%
- Both computed using `jiwer` or equivalent library

**Summarization Quality:**
- **ROUGE-1, ROUGE-2, ROUGE-L** against reference summaries — target: ROUGE-L > 0.45
- **BERTScore** (using HuggingFace model) for semantic similarity — target: F1 > 0.80
- **Entity extraction precision/recall** — did the summary capture the right entities?
- **Schema compliance rate** — % of outputs that pass Pydantic validation without retries

**QA Scoring Accuracy:**
- **Mean Absolute Error (MAE)** between agent scores and human scores — target: MAE < 0.8 per dimension
- **Spearman rank correlation** between agent and human rankings — target: rho > 0.7
- **Compliance flag precision/recall** — false negatives on compliance are critical failures

### 4.3 LLM-as-Judge Evaluation

For scaling evaluation beyond the 25-30 ground truth set:

- Implement a **separate evaluator LLM** (must be a different model than the one generating outputs — if GPT-4o generates, use Claude or Gemini to judge)
- Evaluator scores on these dimensions:
  - **Factual consistency:** Does the summary contain claims not supported by the transcript?
  - **Completeness:** Are key discussion points missing?
  - **Conciseness:** Is there unnecessary filler or repetition?
  - **Actionability:** Are action items specific and assignable?
- Each judgment includes a score (1-5) and a written rationale
- Run via **LangSmith Evaluators** — not ad-hoc scripts

### 4.4 Correlation Analysis

The critical bridge between human and LLM evaluation:

- Run both human and LLM-as-judge on the same 25-30 ground truth calls
- Compute:
  - **Cohen's Kappa** for categorical agreement (resolution status, compliance flags)
  - **Pearson/Spearman correlation** for numerical scores
  - **Confusion matrix** for compliance flag detection
- Document where LLM-as-judge agrees/disagrees with humans and why
- This analysis determines whether LLM-as-judge can be trusted at scale

### 4.5 Evaluation Pipeline (Not Manual)

All evaluation must be automated and reproducible:

```
make eval                  # runs full eval suite
make eval-transcription    # WER + DER only
make eval-summary          # ROUGE + BERTScore + entity metrics
make eval-qa               # MAE + correlation + compliance
make eval-judge            # LLM-as-judge on full dataset
make eval-correlation      # human vs LLM-judge agreement
```

- Results output to `evaluations/results/` as JSON + a summary table
- Integrated into CI — eval suite runs on every PR (on a small subset for speed)
- Full eval suite tracked in LangSmith as experiment runs with comparison across iterations

### 4.6 Minimum Accuracy Thresholds (Gate Criteria)

The system is not considered passing unless:

| Metric | Threshold |
|---|---|
| Transcription WER | < 15% |
| Diarization DER | < 20% |
| Summary ROUGE-L | > 0.45 |
| Summary BERTScore F1 | > 0.80 |
| QA Score MAE | < 0.8 per dimension |
| QA Score Spearman rho | > 0.7 |
| Compliance flag recall | > 0.90 |
| Schema validation pass rate | > 95% |
| LLM-judge / Human Cohen's kappa | > 0.6 |

These are not aspirational — they are pass/fail gates.

---

## 5. Security & Hardening

This section treats the system as if it handles real customer PII. Learners must implement every layer — no "we'd do this in production" hand-waving.

### 5.1 API Key & Secrets Management

- **Zero hardcoded secrets.** No API keys in code, config files, or notebooks. Ever.
- All secrets loaded via environment variables using `python-dotenv` locally, GitHub Secrets in CI/CD, and HuggingFace Spaces secrets in deployment
- `.env` file in `.gitignore` — verified by a pre-commit hook that rejects commits containing key patterns (`sk-`, `OPENAI_API_KEY=`, etc.)
- Provide a `.env.example` with placeholder values documenting every required variable
- **Gate criteria:** Automated scan in CI using `detect-secrets` or `trufflehog` — pipeline fails if secrets are found in any commit

### 5.2 PII Detection & Redaction

- Implement a PII redaction layer that runs at two points:
  1. **Post-transcription** — before transcript is passed to any LLM
  2. **Pre-storage** — before any data is persisted to database
- Detect and redact: phone numbers, email addresses, SSNs, credit card numbers, full names, account numbers, addresses
- Use **Microsoft Presidio** (open source, HuggingFace compatible) or regex-based detection with a named entity model
- Redaction format: `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`, etc. — typed redaction, not generic `[REDACTED]`
- Original unredacted data never touches the database or logs
- **Gate criteria:** Run a test suite with synthetic PII-injected transcripts — 100% of injected PII must be caught. Zero tolerance.

### 5.3 Prompt Injection Defense

- All user-provided inputs (audio filenames, metadata fields, any text input in Gradio) must be sanitized before reaching any LLM prompt
- Implement a prompt injection detection layer before the Summarization and QA Scoring nodes:
  - Scan transcript text for known injection patterns ("ignore previous instructions", "system prompt:", role-switching attempts)
  - Use a classifier-based approach (HuggingFace model fine-tuned for injection detection) or a rules-based filter
  - If injection detected: flag the call, log the attempt to LangSmith, halt processing for that call
- LLM prompts must use parameterized templates — transcript content goes into clearly delimited user message sections, never concatenated into system prompts
- **Gate criteria:** Include an adversarial test set (10-15 transcripts with embedded injection attempts) — all must be detected and blocked

### 5.4 Input Validation & Rate Limiting

- **Gradio input validation:**
  - File type validation (magic bytes, not just extension)
  - File size limits enforced server-side (not just client-side)
  - Maximum concurrent uploads per session
- **LLM call rate limiting:**
  - Implement token budget tracking per call (log via LangSmith)
  - Set maximum retries per node (prevent infinite retry loops from burning budget)
  - Alert if cost per call exceeds configurable threshold

### 5.5 Access Control & Audit Logging

- **Gradio authentication:** Enable Gradio's built-in auth — the app is not publicly accessible without login
- **Audit log for every call processed:**
  - Who uploaded it (authenticated user)
  - Timestamp of each processing stage
  - Which models were invoked (including fallbacks)
  - Whether PII was detected and redacted
  - Whether any security flags were triggered
  - Final output hash (integrity verification)
- Audit log stored separately from call data — append-only, not editable through the application
- **Gate criteria:** Audit log must be queryable — demonstrate retrieving the full processing history for any given `call_id`

### 5.6 Data Encryption

- **At rest:** SQLite database encrypted using `sqlcipher` or equivalent. PostgreSQL with encrypted columns for sensitive fields.
- **In transit:** All API calls over HTTPS (enforced, not optional). Gradio app served over HTTPS via HuggingFace Spaces (default).
- **Stored audio files:** Encrypted on disk if retained, or deleted after processing (configurable retention policy)
- **Gate criteria:** Demonstrate that raw database files are not readable without the encryption key

---

## 6. Deployment & CI/CD

No localhost demos. The system must be deployed, automated, and observable.

### 6.1 Gradio Application

Full-featured UI, not a toy:
- **Upload tab:** Drag-and-drop audio upload with format/size validation feedback, progress indicator during processing
- **Results tab:** Transcript with speaker labels and timestamps, structured summary, QA scores with justifications, compliance flags highlighted
- **Batch tab:** Upload multiple audio files, process sequentially, view/download results as a batch report
- **History tab:** View previously processed calls from the database, search by call_id or date range
- **Evaluation tab:** Run eval suite from UI, display accuracy metrics and comparison charts

Additional requirements:
- Authentication enabled — no anonymous access
- Error states handled visually — if a node fails, the UI shows which stage failed and why, not a generic error
- Downloadable outputs: JSON report, PDF summary (generated via `reportlab` or `weasyprint`)

### 6.2 HuggingFace Spaces Deployment

- App deployed on HuggingFace Spaces using the Gradio SDK
- `README.md` at repo root includes HF Spaces metadata header:
  ```yaml
  ---
  title: Call Center Intelligence System
  emoji: phone
  colorFrom: blue
  colorTo: green
  sdk: gradio
  sdk_version: "5.x"
  app_file: app.py
  pinned: false
  ---
  ```
- Secrets configured via HF Spaces settings — not in repo
- `pyproject.toml` with pinned dependency versions (no `>=` — exact pins for reproducibility)
- Health check endpoint — Gradio app exposes a status route confirming the app is running and models are loaded

### 6.3 GitHub Repository Structure

```
call-center-intelligence/
├── app.py                          # Gradio application entry point
├── pyproject.toml                  # Dependencies, project metadata
├── Makefile                        # eval, test, lint, format, run commands
├── .env.example                    # Required environment variables
├── .gitignore                      # .env, __pycache__, audio files, db files
├── .pre-commit-config.yaml         # Secret scanning, linting hooks
│
├── src/
│   ├── agents/
│   │   ├── intake.py               # Intake node
│   │   ├── transcription.py        # Whisper STT node
│   │   ├── summarization.py        # Summary generation node
│   │   ├── qa_scoring.py           # QA rubric scoring node
│   │   └── report.py               # Report compilation node
│   ├── graph/
│   │   ├── workflow.py             # LangGraph state machine definition
│   │   ├── state.py                # Pydantic state models
│   │   └── edges.py                # Conditional routing logic
│   ├── security/
│   │   ├── pii_redactor.py         # PII detection and redaction
│   │   ├── injection_detector.py   # Prompt injection defense
│   │   └── audit.py                # Audit logging
│   ├── evaluation/
│   │   ├── metrics.py              # WER, ROUGE, BERTScore, MAE calculations
│   │   ├── llm_judge.py            # LLM-as-judge evaluator
│   │   ├── correlation.py          # Human vs LLM agreement analysis
│   │   └── run_eval.py             # Evaluation pipeline orchestrator
│   ├── database/
│   │   ├── models.py               # SQLAlchemy/ORM models
│   │   └── connection.py           # DB connection with encryption
│   └── utils/
│       ├── audio.py                # Audio format validation, properties
│       └── config.py               # Centralized configuration loading
│
├── tests/
│   ├── unit/
│   │   ├── test_intake.py
│   │   ├── test_transcription.py
│   │   ├── test_summarization.py
│   │   ├── test_qa_scoring.py
│   │   ├── test_pii_redactor.py
│   │   └── test_injection_detector.py
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py
│   │   └── test_database_persistence.py
│   └── security/
│       ├── test_pii_detection.py       # Synthetic PII injection tests
│       └── test_prompt_injection.py    # Adversarial injection tests
│
├── evaluations/
│   ├── ground_truth/                   # 25-30 annotated call JSONs
│   ├── adversarial/                    # Prompt injection test transcripts
│   ├── results/                        # Eval run outputs (gitignored)
│   └── config.yaml                     # Eval thresholds and settings
│
├── data/
│   └── README.md                       # Instructions for downloading datasets
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                     # Lint + unit tests + secret scan on every PR
│       ├── eval.yaml                   # Eval suite on main branch pushes
│       └── deploy.yaml                 # Auto-deploy to HF Spaces on main merge
│
└── docs/
    ├── architecture.md                 # System design document
    ├── evaluation_report.md            # Eval results and analysis
    └── security.md                     # Security posture documentation
```

### 6.4 CI/CD Pipeline (GitHub Actions)

**On every Pull Request (`ci.yaml`):**
1. **Lint:** `ruff check` + `ruff format --check`
2. **Type check:** `mypy` on `src/`
3. **Secret scan:** `detect-secrets` — fails if secrets found
4. **Unit tests:** `pytest tests/unit/ -v`
5. **Security tests:** `pytest tests/security/ -v`
6. All must pass before merge is allowed (branch protection rules enabled)

**On merge to main (`eval.yaml`):**
1. Run full unit + integration test suite
2. Run evaluation pipeline on ground truth set (subset for speed)
3. Post eval results as a comment on the merge commit or as a GitHub Actions summary
4. Fail if any accuracy metric drops below threshold

**On merge to main (`deploy.yaml`):**
1. Triggered after `eval.yaml` passes
2. Push to HuggingFace Spaces via `huggingface_hub` CLI
3. Verify health check endpoint responds after deployment

### 6.5 Observability via LangSmith

- **Every LLM call traced** — input, output, latency, token count, cost
- **Every pipeline run** is a LangSmith session — can replay and inspect any call's full processing
- **Evaluation runs** tracked as LangSmith experiments — compare accuracy across code changes
- **Dashboard:** Learners must demonstrate a LangSmith dashboard showing:
  - Average latency per node
  - Token usage and cost per call
  - Error rate per node
  - Eval metric trends over time

---

## 7. Milestones & Gate Criteria

Five milestones with hard pass/fail gates. No partial credit on gates — either the system meets the bar or it doesn't.

### Milestone 1: Data & Transcription Pipeline

**Deliverables:**
- Curated dataset of minimum 50 real call center audio files downloaded and organized
- Intake Agent with full validation (format, size, duration, metadata)
- Transcription Agent using locally-hosted Whisper model (not API) with speaker diarization
- Pydantic models for `IntakeResult` and `TranscriptionResult`
- Unit tests for intake validation (valid/invalid formats, oversized files, corrupt audio)
- Unit tests for transcription output structure

**Gate Criteria:**
- [ ] Intake rejects invalid audio with specific error messages
- [ ] Whisper transcription runs end-to-end on at least 50 audio files
- [ ] Speaker diarization distinguishes minimum 2 speakers
- [ ] Confidence scores produced per segment
- [ ] All Pydantic models validate correctly
- [ ] Unit tests pass

### Milestone 2: Multi-Agent System

**Deliverables:**
- Summarization Agent producing structured output via function calling
- QA Scoring Agent with 5-dimension rubric and justifications
- Report Generation Agent compiling all results
- Full LangGraph workflow connecting all nodes with conditional edges
- Fallback LLM routing (primary fails → fallback model)
- Retry logic on malformed LLM outputs
- LangSmith tracing on every node
- Integration test: audio file in → full CallReport out

**Gate Criteria:**
- [ ] Full pipeline processes an audio file end-to-end without manual intervention
- [ ] Summary follows defined schema — no free-form outputs
- [ ] QA scores include text justifications citing transcript segments
- [ ] Fallback model activates when primary is unavailable (test by simulating failure)
- [ ] All outputs are Pydantic-validated
- [ ] LangSmith traces visible for every pipeline run
- [ ] Integration test passes

### Milestone 3: Evaluation & Accuracy

**Deliverables:**
- 25-30 calls with human-annotated ground truth (reference summaries, QA scores, corrected transcripts)
- Automated evaluation pipeline (`make eval`) computing all metrics
- LLM-as-judge evaluator using a different model than the primary
- Correlation analysis between human and LLM-judge scores
- Evaluation results documented in `docs/evaluation_report.md`
- Eval pipeline integrated into CI

**Gate Criteria:**
- [ ] WER < 15% on ground truth transcripts
- [ ] Summary ROUGE-L > 0.45 and BERTScore F1 > 0.80
- [ ] QA Score MAE < 0.8 per dimension
- [ ] QA Score Spearman rho > 0.7
- [ ] Compliance flag recall > 0.90
- [ ] Schema validation pass rate > 95%
- [ ] LLM-judge / Human Cohen's kappa > 0.6
- [ ] `make eval` runs without errors and outputs results to `evaluations/results/`
- [ ] Evaluation report explains where the system fails and why

### Milestone 4: Security & Hardening

**Deliverables:**
- PII redaction layer with typed redaction tags
- Prompt injection detection and blocking
- Input validation (magic bytes, server-side size limits)
- Gradio authentication enabled
- Audit logging for every processed call
- Database encryption (SQLCipher or equivalent)
- Pre-commit hook for secret scanning
- Security test suite (PII injection tests, adversarial prompt tests)

**Gate Criteria:**
- [ ] 100% of synthetic PII caught in test suite (zero misses)
- [ ] All adversarial injection attempts detected and blocked
- [ ] No secrets in any commit (verified by `detect-secrets`)
- [ ] `.env.example` exists, `.env` is gitignored
- [ ] Audit log is queryable — demonstrate full history retrieval for a call_id
- [ ] Database file is not readable without encryption key
- [ ] Gradio app requires authentication
- [ ] Security tests pass in CI

### Milestone 5: Production Deployment

**Deliverables:**
- Gradio app with all tabs (Upload, Results, Batch, History, Evaluation)
- Deployed on HuggingFace Spaces — publicly accessible with auth
- GitHub Actions CI/CD: lint + test + eval + deploy
- Branch protection rules enabled on main
- LangSmith dashboard with latency, cost, error rate, eval trends
- Complete documentation (architecture, eval report, security posture)
- README with setup instructions, dataset download steps, environment configuration

**Gate Criteria:**
- [ ] App is live on HuggingFace Spaces and functional
- [ ] CI pipeline passes on a fresh PR (lint + tests + secret scan)
- [ ] Eval pipeline runs in CI and posts results
- [ ] Deployment auto-triggers on merge to main
- [ ] LangSmith dashboard shows real data from processed calls
- [ ] README enables a new developer to set up and run the project from scratch
- [ ] End-to-end demo: upload audio → view transcript, summary, QA scores, download report

---

## 8. Grading Rubric

| Category | Weight | What Is Evaluated |
|---|---|---|
| **Agent Accuracy & Evaluation** | 30% | Ground truth annotation quality, metric implementation, threshold achievement, correlation analysis, eval pipeline automation |
| **Agent Architecture** | 25% | LangGraph state machine design, typed contracts, routing logic, fallback handling, retry mechanisms, LangSmith integration |
| **Security Posture** | 20% | PII redaction coverage, injection defense, audit logging, encryption, secret management, security test suite |
| **Deployment & CI/CD** | 15% | Working HF Spaces deployment, GitHub Actions pipeline, branch protection, automated eval in CI, health checks |
| **UI & Documentation** | 10% | Gradio app completeness, error state handling, README quality, architecture doc, eval report |

**Weighting notes:**
- Accuracy & Evaluation is the single largest category — a beautiful app with inaccurate outputs fails
- Security is weighted higher than deployment — a deployed insecure system is worse than an undeployed secure one
- Documentation is 10% but is a hard gate — missing README or eval report is an automatic incomplete

---

## 9. Curated Dataset Sources

Learners must use real, publicly available call center or conversational audio datasets. Approved sources:

1. **LibriSpeech** (subset) — clean speech for baseline transcription testing
   - Source: openslr.org/12
2. **Common Voice by Mozilla** — diverse accents and recording conditions
   - Source: commonvoice.mozilla.org
3. **CallHome** (LDC) — real telephone conversations (if institutional access available)
   - Source: catalog.ldc.upenn.edu
4. **VoxForge** — open-source speech corpus
   - Source: voxforge.org
5. **CCCS (Call Centre Customer Service) datasets on HuggingFace** — search HuggingFace Datasets hub for call center / customer service tagged datasets

Learners must document which dataset(s) they used, how many files, total audio hours, and any preprocessing applied.

---

## 10. Submission Requirements

### Prototype Demonstration
- Upload a call audio sample through the live Gradio app
- Generate structured summary and QA score end-to-end
- Show all Gradio tabs functional: Upload, Results, Batch, History, Evaluation

### Demo Video
- Showcase end-to-end flow: Audio upload → Transcription → Summary → QA Scores → Report download
- Demonstrate fallback logic (simulate primary model failure)
- Show LangSmith traces for a processed call
- Show security features: PII redaction in action, authentication, audit log query
- Show eval pipeline execution and results

### GitHub Repository
- Public repo with all code, tests, CI/CD workflows, documentation
- Clean commit history demonstrating incremental development
- All gate criteria met and verifiable by running `make eval` and `make test`

### Documentation Package
- `README.md` — setup, dataset download, environment config, running locally
- `docs/architecture.md` — system design decisions and trade-offs
- `docs/evaluation_report.md` — full eval results, analysis, failure cases
- `docs/security.md` — security posture, threat model, mitigation strategies
