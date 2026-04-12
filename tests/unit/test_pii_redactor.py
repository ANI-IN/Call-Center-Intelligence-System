from src.security.pii_redactor import redact_pii


class TestRedactPII:
    def test_redacts_phone_number(self) -> None:
        text = "Call me at 555-123-4567 please."
        result = redact_pii(text)
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "555-123-4567" not in result.redacted_text
        assert result.pii_found is True

    def test_redacts_email(self) -> None:
        text = "Send it to john.doe@example.com"
        result = redact_pii(text)
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text

    def test_redacts_ssn(self) -> None:
        text = "My SSN is 123-45-6789."
        result = redact_pii(text)
        assert "[REDACTED_SSN]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text

    def test_redacts_credit_card(self) -> None:
        text = "Card number is 4111 1111 1111 1111"
        result = redact_pii(text)
        assert "[REDACTED_CREDIT_CARD]" in result.redacted_text
        assert "4111 1111 1111 1111" not in result.redacted_text

    def test_no_pii_returns_original(self) -> None:
        text = "Hello, how can I help you today?"
        result = redact_pii(text)
        assert result.redacted_text == text
        assert result.pii_found is False

    def test_redacts_multiple_pii_types(self) -> None:
        text = "Call 555-123-4567 or email john@test.com, SSN 123-45-6789"
        result = redact_pii(text)
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert "[REDACTED_SSN]" in result.redacted_text
        assert len(result.detections) == 3
