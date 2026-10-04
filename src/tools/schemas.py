"""Strict Pydantic contracts for every LLM tool call (inputs) and result (outputs).

Inputs use ``strict=True`` so JSON coming back from the model is never coerced
(``"1001"`` is rejected for an ``int`` field), and ``extra="forbid"`` so the
model cannot smuggle unexpected parameters into a handler.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints

OrderId = Annotated[int, Field(gt=0, lt=2**31, description="Numeric order ID, e.g. 1001")]
Reasoning = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=10, max_length=2000),
    Field(description="Short justification for this action. Stored verbatim in the audit log."),
]


class _StrictInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------- inputs


class FetchOrderDetailsInput(_StrictInput):
    """Look up a single order the sender owns."""

    order_id: OrderId


class IssueRefundInput(_StrictInput):
    """Request a full refund for an order. The server re-validates eligibility."""

    order_id: OrderId
    reasoning: Reasoning


# -------------------------------------------------------------------- outputs


class LookupStatus(str, enum.Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"


class OrderDetails(_Output):
    order_id: int
    amount: Decimal
    purchase_date: datetime
    refund_status: str
    days_since_purchase: int = Field(description="Whole days, computed server-side")
    within_refund_window: bool = Field(description="Computed server-side; authoritative")


class FetchOrderDetailsOutput(_Output):
    status: LookupStatus
    order: OrderDetails | None = None
    message: str


class RefundOutcome(str, enum.Enum):
    REFUNDED = "refunded"
    ALREADY_REFUNDED = "already_refunded"
    OUTSIDE_POLICY_WINDOW = "outside_policy_window"
    ORDER_NOT_FOUND = "order_not_found"


class IssueRefundOutput(_Output):
    outcome: RefundOutcome
    order_id: int
    refunded_amount: Decimal | None = None
    audit_log_id: int | None = None
    message: str


class ToolError(_Output):
    error: str
    details: list[dict] | None = None
