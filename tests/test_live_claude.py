"""Optional smoke tests against the real Claude API.

Skipped by default. Run with:  RUN_LIVE_TESTS=1 pytest -m live
"""

from __future__ import annotations

import os

import pytest

from src.agent import Decision, RefundAgent
from tests.conftest import fixed_clock, raw_email

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]


@pytest.mark.parametrize(
    "sender, body, expected",
    [
        ("alice@example.com", "My mug from order 1001 arrived cracked. Refund please.", Decision.REFUNDED),
        ("bob@example.com", "I'd like a refund for order 1002.", Decision.REJECTED_OUTSIDE_POLICY),
        ("carol@example.com", "Refund order 1003 please.", Decision.REJECTED_ALREADY_REFUNDED),
    ],
)
def test_live_decisions(sessions, sender, body, expected):
    result = RefundAgent(sessions, clock=fixed_clock).handle_email(raw_email(sender, body))
    assert result.decision == expected, result
    assert result.reply_email
