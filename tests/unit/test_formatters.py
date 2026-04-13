from src.utils.formatters import format_qa, format_summary, secs_to_mmss


class TestSecsToMmss:
    def test_zero(self) -> None:
        assert secs_to_mmss(0.0) == "00:00"

    def test_one_minute_thirty(self) -> None:
        assert secs_to_mmss(90.0) == "01:30"

    def test_over_ten_minutes(self) -> None:
        assert secs_to_mmss(630.5) == "10:30"


class TestFormatSummary:
    def test_includes_call_purpose(self) -> None:
        data = {
            "call_purpose": "Billing dispute",
            "key_discussion_points": [],
            "action_items": [],
            "resolution_status": "resolved",
            "sentiment_trajectory": "Neutral",
        }
        result = format_summary(data)
        assert "Billing dispute" in result

    def test_includes_action_items(self) -> None:
        data = {
            "call_purpose": "Test",
            "key_discussion_points": [],
            "action_items": [
                {"owner": "agent", "description": "Refund $50", "deadline": "2026-04-15"}
            ],
            "resolution_status": "resolved",
            "sentiment_trajectory": "Neutral",
        }
        result = format_summary(data)
        assert "Refund $50" in result
        assert "Agent" in result


class TestFormatQa:
    def test_includes_overall_score(self) -> None:
        data = {
            "overall_score": 4,
            "professionalism": {"score": 4, "justification": "Good"},
            "empathy": {"score": 4, "justification": "Good"},
            "problem_resolution": {"score": 4, "justification": "Good"},
            "compliance": {"score": 4, "justification": "Good"},
            "communication_clarity": {"score": 4, "justification": "Good"},
            "compliance_flags": [],
        }
        result = format_qa(data)
        assert "4/5" in result
        assert "No compliance issues" in result

    def test_includes_dimension_scores(self) -> None:
        data = {
            "overall_score": 3,
            "professionalism": {"score": 3, "justification": "Adequate."},
            "empathy": {"score": 4, "justification": "Good"},
            "problem_resolution": {"score": 3, "justification": "OK"},
            "compliance": {"score": 3, "justification": "OK"},
            "communication_clarity": {"score": 3, "justification": "OK"},
        }
        result = format_qa(data)
        assert "Professionalism: 3/5" in result
        assert "Empathy: 4/5" in result

    def test_includes_compliance_flags_with_icons(self) -> None:
        data = {
            "overall_score": 3,
            "professionalism": {"score": 3, "justification": "OK"},
            "empathy": {"score": 3, "justification": "OK"},
            "problem_resolution": {"score": 3, "justification": "OK"},
            "compliance": {"score": 3, "justification": "OK"},
            "communication_clarity": {"score": 3, "justification": "OK"},
            "compliance_flags": [
                {"severity": "high", "violation": "No ID check", "transcript_reference": "01:30"}
            ],
        }
        result = format_qa(data)
        assert "No ID check" in result
        assert "HIGH" in result
