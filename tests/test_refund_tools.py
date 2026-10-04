"""Tool-level tests: business rules, idempotency, ACID behaviour, input validation."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from src.database import AuditAction, AuditLog, Order, RefundStatus
from src.tools import ToolContext, build_registry, is_within_refund_window
from src.tools.refund_tools import PolicyViolation
from tests.conftest import FIXED_NOW, fixed_clock

REASON = "Customer reports the item arrived damaged and requests a refund."


@pytest.fixture
def registry():
    return build_registry()


def ctx(sessions, sender):
    return ToolContext(session_factory=sessions, sender_email=sender, clock=fixed_clock)


def refund(registry, sessions, order_id, sender, reasoning=REASON):
    return registry.execute("issue_refund", {"order_id": order_id, "reasoning": reasoning},
                            ctx(sessions, sender))


def order_status(sessions, order_id) -> RefundStatus:
    with sessions() as s:
        return s.get(Order, order_id).refund_status


def audit_rows(sessions, order_id=None) -> list[AuditLog]:
    with sessions() as s:
        q = select(AuditLog).order_by(AuditLog.id)
        if order_id is not None:
            q = q.where(AuditLog.order_id == order_id)
        return list(s.scalars(q))


# ------------------------------------------------------------ happy path


def test_successful_refund_updates_order_and_writes_audit(registry, sessions):
    res = refund(registry, sessions, 1001, "alice@example.com")

    assert not res.is_error
    payload = json.loads(res.content)
    assert payload["outcome"] == "refunded"
    assert Decimal(payload["refunded_amount"]) == Decimal("49.99")
    assert order_status(sessions, 1001) == RefundStatus.REFUNDED

    [row] = audit_rows(sessions, 1001)
    assert row.action_taken == AuditAction.REFUND_ISSUED
    assert row.llm_reasoning == REASON
    assert row.timestamp == FIXED_NOW
    assert payload["audit_log_id"] == row.id


def test_sender_email_match_is_case_insensitive(registry, sessions):
    res = refund(registry, sessions, 1001, "ALICE@Example.COM")
    assert json.loads(res.content)["outcome"] == "refunded"


# ------------------------------------------------------- 30-day policy


def test_rejects_refund_older_than_30_days(registry, sessions):
    res = refund(registry, sessions, 1002, "bob@example.com")

    assert json.loads(res.content)["outcome"] == "outside_policy_window"
    assert order_status(sessions, 1002) == RefundStatus.NONE
    [row] = audit_rows(sessions, 1002)
    assert row.action_taken == AuditAction.REFUND_REJECTED_OUTSIDE_POLICY


@pytest.mark.parametrize(
    "age, expected",
    [
        (timedelta(days=29, hours=23), True),
        (timedelta(days=30), True),  # exactly 30 days still qualifies
        (timedelta(days=30, seconds=1), False),
        (timedelta(days=31), False),
        (timedelta(0), True),
    ],
)
def test_refund_window_boundaries(age, expected):
    assert is_within_refund_window(FIXED_NOW - age, FIXED_NOW) is expected


def test_window_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        is_within_refund_window(FIXED_NOW.replace(tzinfo=None), FIXED_NOW)


def test_window_rejects_future_purchase_date():
    with pytest.raises(PolicyViolation):
        is_within_refund_window(FIXED_NOW + timedelta(days=1), FIXED_NOW)


def test_boundary_orders_from_seed(registry, sessions):
    assert json.loads(refund(registry, sessions, 1004, "dave@example.com").content)["outcome"] == "refunded"
    assert json.loads(refund(registry, sessions, 1005, "erin@example.com").content)["outcome"] == "outside_policy_window"


def test_llm_cannot_assert_eligibility(registry, sessions):
    """The model can't supply its own date math or policy verdict; extra fields are rejected."""
    res = registry.execute(
        "issue_refund",
        {"order_id": 1002, "reasoning": REASON, "within_refund_window": True, "days_since_purchase": 3},
        ctx(sessions, "bob@example.com"),
    )
    assert res.is_error
    assert order_status(sessions, 1002) == RefundStatus.NONE
    assert audit_rows(sessions) == []


# ---------------------------------------------------------- idempotency


def test_rejects_already_refunded_order(registry, sessions):
    res = refund(registry, sessions, 1003, "carol@example.com")

    assert json.loads(res.content)["outcome"] == "already_refunded"
    [row] = audit_rows(sessions, 1003)
    assert row.action_taken == AuditAction.REFUND_REJECTED_ALREADY_REFUNDED


def test_repeated_refund_calls_refund_exactly_once(registry, sessions):
    outcomes = [json.loads(refund(registry, sessions, 1001, "alice@example.com").content)["outcome"]
                for _ in range(3)]

    assert outcomes == ["refunded", "already_refunded", "already_refunded"]
    issued = [r for r in audit_rows(sessions, 1001) if r.action_taken == AuditAction.REFUND_ISSUED]
    assert len(issued) == 1


def test_concurrent_refunds_are_serialized(registry, sessions):
    """Eight threads race to refund the same order; BEGIN IMMEDIATE + CAS allow exactly one."""
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        res = refund(registry, sessions, 1001, "alice@example.com")
        with lock:
            outcomes.append(json.loads(res.content).get("outcome", "error"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("refunded") == 1
    assert outcomes.count("already_refunded") == 7
    issued = [r for r in audit_rows(sessions, 1001) if r.action_taken == AuditAction.REFUND_ISSUED]
    assert len(issued) == 1


# ------------------------------------------------------------- atomicity


def test_audit_failure_rolls_back_refund(registry, sessions, monkeypatch):
    """If the audit write fails, the order status change must not persist."""
    import src.tools.refund_tools as rt

    real_audit = rt.AuditLog

    def exploding_audit(**kwargs):
        if kwargs["action_taken"] == AuditAction.REFUND_ISSUED:
            raise RuntimeError("disk full")
        return real_audit(**kwargs)

    monkeypatch.setattr(rt, "AuditLog", exploding_audit)

    res = refund(registry, sessions, 1001, "alice@example.com")

    assert res.is_error
    assert "no changes were made" in json.loads(res.content)["error"]
    assert order_status(sessions, 1001) == RefundStatus.NONE
    assert audit_rows(sessions) == []


def test_db_constraint_failure_rolls_back(registry, sessions, monkeypatch):
    """A failure raised by the database itself (NOT NULL violation) also rolls back."""
    import src.tools.refund_tools as rt

    real_audit = rt.AuditLog

    def null_reasoning_audit(**kwargs):
        if kwargs["action_taken"] == AuditAction.REFUND_ISSUED:
            kwargs["llm_reasoning"] = None
        return real_audit(**kwargs)

    monkeypatch.setattr(rt, "AuditLog", null_reasoning_audit)

    res = refund(registry, sessions, 1001, "alice@example.com")

    assert res.is_error
    assert order_status(sessions, 1001) == RefundStatus.NONE
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(AuditLog)) == 0


# ------------------------------------------------------------ ownership


def test_cannot_refund_someone_elses_order(registry, sessions):
    res = refund(registry, sessions, 1001, "mallory@example.com")

    assert json.loads(res.content)["outcome"] == "order_not_found"
    assert order_status(sessions, 1001) == RefundStatus.NONE
    assert audit_rows(sessions) == []


def test_fetch_hides_other_customers_orders(registry, sessions):
    res = registry.execute("fetch_order_details", {"order_id": 1001},
                           ctx(sessions, "mallory@example.com"))
    assert json.loads(res.content)["status"] == "not_found"


def test_fetch_reports_server_computed_window(registry, sessions):
    res = registry.execute("fetch_order_details", {"order_id": 1002}, ctx(sessions, "bob@example.com"))
    order = json.loads(res.content)["order"]
    assert order["days_since_purchase"] == 45
    assert order["within_refund_window"] is False


# ------------------------------------------------------ input validation


@pytest.mark.parametrize(
    "bad_input",
    [
        {"order_id": "1001", "reasoning": REASON},  # string instead of int (no coercion)
        {"order_id": 1001.0, "reasoning": REASON},  # float
        {"order_id": -5, "reasoning": REASON},
        {"order_id": 0, "reasoning": REASON},
        {"order_id": True, "reasoning": REASON},
        {"order_id": 1001},  # missing reasoning
        {"order_id": 1001, "reasoning": "ok"},  # too short
        {"order_id": 1001, "reasoning": "x" * 2001},
        {"order_id": "1001; DROP TABLE orders", "reasoning": REASON},
        "refund order 1001 please",  # raw string, not an object
        None,
    ],
)
def test_issue_refund_rejects_invalid_input(registry, sessions, bad_input):
    res = registry.execute("issue_refund", bad_input, ctx(sessions, "alice@example.com"))

    assert res.is_error
    assert json.loads(res.content)["error"] == "Invalid tool input"
    assert order_status(sessions, 1001) == RefundStatus.NONE
    assert audit_rows(sessions) == []


def test_unknown_tool_is_an_error(registry, sessions):
    res = registry.execute("delete_all_orders", {}, ctx(sessions, "alice@example.com"))
    assert res.is_error


def test_tool_definitions_are_closed_schemas(registry):
    defs = {d["name"]: d for d in registry.definitions()}
    assert set(defs) == {"fetch_order_details", "issue_refund"}
    for d in defs.values():
        assert d["input_schema"]["type"] == "object"
        assert d["input_schema"]["additionalProperties"] is False
    assert defs["issue_refund"]["input_schema"]["required"] == ["order_id", "reasoning"]
    assert defs["issue_refund"]["input_schema"]["properties"]["order_id"]["type"] == "integer"
