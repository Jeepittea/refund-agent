from __future__ import annotations

import pytest

from src.agent.email_parser import MalformedEmailError, parse_raw_email
from tests.conftest import raw_email


def test_parses_valid_email():
    email = parse_raw_email(raw_email("Alice@Example.com", "Please refund order 1001."))
    assert email.sender == "alice@example.com"
    assert email.subject == "Refund request"
    assert email.body == "Please refund order 1001."


def test_parses_multipart_email_text_part():
    raw = (
        "From: alice@example.com\nSubject: hi\nMIME-Version: 1.0\n"
        'Content-Type: multipart/alternative; boundary="b"\n\n'
        "--b\nContent-Type: text/plain\n\nRefund order 1001\n"
        "--b\nContent-Type: text/html\n\n<p>Refund order 1001</p>\n--b--\n"
    )
    assert parse_raw_email(raw).body == "Refund order 1001"


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("hey can i get my money back for order 1001", "no headers"),
        ("Subject: refund\n\nRefund order 1001", "no From"),
        ("From: not-an-email\nSubject: refund\n\nRefund order 1001", "invalid From"),
        ("From: alice@\nSubject: refund\n\nRefund order 1001", "invalid From"),
        ("From: alice@example.com\nSubject: refund\n\n   \n", "empty body"),
        ("From: alice@example.com\nSubject: refund\n\n" + "x" * 200_000, "oversize"),
        (
            "From: alice@example.com\nContent-Type: text/html\n\n<p>Refund order 1001</p>",
            "no text/plain body",
        ),
    ],
    # Name cases by their reason; the raw inputs are too long/multiline for test IDs.
    ids=lambda v: v.replace(" ", "-") if len(v) < 25 and "\n" not in v else "raw",
)
def test_rejects_malformed_email(raw, reason):
    with pytest.raises(MalformedEmailError):
        parse_raw_email(raw)


def test_rejects_non_string():
    with pytest.raises(MalformedEmailError):
        parse_raw_email(None)  # type: ignore[arg-type]
