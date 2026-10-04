"""Business-logic tools. All policy decisions are made here, in Python, never by the LLM."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select, update

from src.database.models import AuditAction, AuditLog, Order, RefundStatus
from src.tools.registry import ToolContext, ToolRegistry, ToolSpec
from src.tools.schemas import (
    FetchOrderDetailsInput,
    FetchOrderDetailsOutput,
    IssueRefundInput,
    IssueRefundOutput,
    LookupStatus,
    OrderDetails,
    RefundOutcome,
)

REFUND_WINDOW = timedelta(days=30)


class PolicyViolation(Exception):
    pass


def is_within_refund_window(purchase_date: datetime, now: datetime) -> bool:
    """True iff ``now`` is no more than 30 days (to the second) after purchase.

    A purchase date in the future is a data-integrity problem, not a free pass.
    """
    if purchase_date.tzinfo is None or now.tzinfo is None:
        raise ValueError("Both datetimes must be timezone-aware")
    elapsed = now - purchase_date
    if elapsed < timedelta(0):
        raise PolicyViolation(f"purchase_date {purchase_date.isoformat()} is in the future")
    return elapsed <= REFUND_WINDOW


def _owned_order(session, order_id: int, sender_email: str) -> Order | None:
    """Fetch an order only if it belongs to the sender.

    Orders owned by someone else are reported as not found, so the tool can't be
    used to check whether an order ID exists.
    """
    order = session.get(Order, order_id)
    if order is None or order.customer_email.casefold() != sender_email.casefold():
        return None
    return order


# ------------------------------------------------------------------ fetch tool


def fetch_order_details(args: FetchOrderDetailsInput, ctx: ToolContext) -> FetchOrderDetailsOutput:
    now = ctx.clock()
    with ctx.session_factory() as session:
        order = _owned_order(session, args.order_id, ctx.sender_email)
        if order is None:
            return FetchOrderDetailsOutput(
                status=LookupStatus.NOT_FOUND,
                message=f"No order {args.order_id} found for the sender's email address.",
            )
        within = is_within_refund_window(order.purchase_date, now)
        return FetchOrderDetailsOutput(
            status=LookupStatus.FOUND,
            order=OrderDetails(
                order_id=order.id,
                amount=order.amount,
                purchase_date=order.purchase_date,
                refund_status=order.refund_status.value,
                days_since_purchase=(now - order.purchase_date).days,
                within_refund_window=within,
            ),
            message="Order found. within_refund_window is authoritative.",
        )


# ----------------------------------------------------------------- refund tool


def issue_refund(args: IssueRefundInput, ctx: ToolContext) -> IssueRefundOutput:
    """Idempotent, ACID refund.

    Everything below happens in ONE transaction opened with BEGIN IMMEDIATE
    (see ``make_engine``), so the checks and the write can't interleave with
    another refund attempt. The status flip is also a compare-and-set
    (``WHERE refund_status = 'none'``); if it touches zero rows we treat it as
    already refunded. Any exception rolls back both the order update and the
    audit row.
    """
    now = ctx.clock()
    with ctx.session_factory.begin() as session:
        order = _owned_order(session, args.order_id, ctx.sender_email)
        if order is None:
            return IssueRefundOutput(
                outcome=RefundOutcome.ORDER_NOT_FOUND,
                order_id=args.order_id,
                message=f"No order {args.order_id} found for the sender's email address.",
            )

        def audit(action: AuditAction) -> AuditLog:
            entry = AuditLog(order_id=order.id, action_taken=action,
                             llm_reasoning=args.reasoning, timestamp=now)
            session.add(entry)
            session.flush()
            return entry

        # 1. Idempotency check
        if order.refund_status == RefundStatus.REFUNDED:
            entry = audit(AuditAction.REFUND_REJECTED_ALREADY_REFUNDED)
            return IssueRefundOutput(
                outcome=RefundOutcome.ALREADY_REFUNDED, order_id=order.id,
                audit_log_id=entry.id,
                message="This order has already been refunded. No new refund was issued.",
            )

        # 2. Policy check, deterministic Python datetime math
        if not is_within_refund_window(order.purchase_date, now):
            entry = audit(AuditAction.REFUND_REJECTED_OUTSIDE_POLICY)
            return IssueRefundOutput(
                outcome=RefundOutcome.OUTSIDE_POLICY_WINDOW, order_id=order.id,
                audit_log_id=entry.id,
                message=f"Purchase was {(now - order.purchase_date).days} days ago; "
                        "refunds are limited to 30 days. No refund was issued.",
            )

        # 3. Compare-and-set + audit write, committed together
        result = session.execute(
            update(Order)
            .where(Order.id == order.id, Order.refund_status == RefundStatus.NONE)
            .values(refund_status=RefundStatus.REFUNDED)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            entry = audit(AuditAction.REFUND_REJECTED_ALREADY_REFUNDED)
            return IssueRefundOutput(
                outcome=RefundOutcome.ALREADY_REFUNDED, order_id=order.id,
                audit_log_id=entry.id,
                message="This order has already been refunded. No new refund was issued.",
            )
        entry = audit(AuditAction.REFUND_ISSUED)
        # Leaving the `with` block commits; an exception anywhere above rolls back.
        return IssueRefundOutput(
            outcome=RefundOutcome.REFUNDED, order_id=order.id,
            refunded_amount=order.amount, audit_log_id=entry.id,
            message=f"Refund of {order.amount} issued.",
        )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="fetch_order_details",
        description=(
            "Look up an order belonging to the customer who sent the email. Returns amount, "
            "purchase date, refund status, and a server-computed `within_refund_window` flag. "
            "Call this before deciding anything about a refund."
        ),
        input_model=FetchOrderDetailsInput,
        output_model=FetchOrderDetailsOutput,
        handler=fetch_order_details,
    ))
    registry.register(ToolSpec(
        name="issue_refund",
        description=(
            "Issue a full refund for an order owned by the sender. The server re-checks "
            "ownership, the 30-day window and prior refunds, and is idempotent: calling it "
            "for an already-refunded order never refunds twice. Read the returned `outcome`; "
            "only `refunded` means money was returned."
        ),
        input_model=IssueRefundInput,
        output_model=IssueRefundOutput,
        handler=issue_refund,
    ))
    return registry
