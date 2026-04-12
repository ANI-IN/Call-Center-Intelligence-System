import pytest

from src.security.pii_redactor import redact_pii


@pytest.mark.security
class TestPIIDetectionComprehensive:
    @pytest.mark.parametrize(
        "phone",
        [
            "555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "5551234567",
            "+1-555-123-4567",
        ],
    )
    def test_catches_phone_formats(self, phone: str) -> None:
        result = redact_pii(f"Number: {phone}")
        assert phone not in result.redacted_text, f"Failed to redact phone: {phone}"

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "first.last@company.co.uk",
            "user+tag@gmail.com",
        ],
    )
    def test_catches_email_formats(self, email: str) -> None:
        result = redact_pii(f"Email: {email}")
        assert email not in result.redacted_text, f"Failed to redact email: {email}"

    @pytest.mark.parametrize(
        "ssn",
        [
            "123-45-6789",
            "123 45 6789",
        ],
    )
    def test_catches_ssn_formats(self, ssn: str) -> None:
        result = redact_pii(f"SSN: {ssn}")
        assert ssn not in result.redacted_text, f"Failed to redact SSN: {ssn}"

    @pytest.mark.parametrize(
        "cc",
        [
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "5500000000000004",
        ],
    )
    def test_catches_credit_card_formats(self, cc: str) -> None:
        result = redact_pii(f"Card: {cc}")
        assert cc not in result.redacted_text, f"Failed to redact CC: {cc}"

    def test_embedded_pii_in_conversation(self) -> None:
        transcript = (
            "Agent: Can you verify your identity?\n"
            "Customer: Sure, my social is 234-56-7890 and "
            "you can reach me at jane@company.com or 415-555-0199.\n"
            "My card ending in 4242 4242 4242 4242."
        )
        result = redact_pii(transcript)
        assert "234-56-7890" not in result.redacted_text
        assert "jane@company.com" not in result.redacted_text
        assert "415-555-0199" not in result.redacted_text
        assert "4242 4242 4242 4242" not in result.redacted_text
