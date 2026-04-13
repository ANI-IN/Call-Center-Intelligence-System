import uuid

from src.agents.report import compile_report, generate_report_json, generate_report_pdf
from tests.conftest import make_intake_result, make_qa_scores, make_summary, make_transcription


class TestCompileReport:
    def test_creates_valid_report(self) -> None:
        cid = uuid.uuid4()
        report = compile_report(
            intake=make_intake_result(call_id=cid),
            transcription=make_transcription(call_id=cid),
            summary=make_summary(call_id=cid),
            qa_scores=make_qa_scores(call_id=cid),
            trace_id="test-trace",
        )
        assert report.call_id == cid
        assert report.status == "completed"
        assert report.trace_id == "test-trace"


class TestGenerateReportJson:
    def test_produces_valid_json(self) -> None:
        cid = uuid.uuid4()
        report = compile_report(
            intake=make_intake_result(call_id=cid),
            transcription=make_transcription(call_id=cid),
            summary=make_summary(call_id=cid),
            qa_scores=make_qa_scores(call_id=cid),
            trace_id="",
        )
        json_str = generate_report_json(report)
        assert '"call_id"' in json_str
        assert '"summary"' in json_str


class TestGenerateReportPdf:
    def test_produces_non_empty_pdf(self) -> None:
        cid = uuid.uuid4()
        report = compile_report(
            intake=make_intake_result(call_id=cid),
            transcription=make_transcription(call_id=cid),
            summary=make_summary(call_id=cid),
            qa_scores=make_qa_scores(call_id=cid),
            trace_id="",
        )
        pdf_bytes = generate_report_pdf(report)
        assert len(pdf_bytes) > 100
        assert pdf_bytes[:5] == b"%PDF-"
