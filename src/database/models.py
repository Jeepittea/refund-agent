"""SQLAlchemy ORM models for orders and the append-only audit log."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator):
    """Stores datetimes as naive UTC, always returns timezone-aware UTC.

    SQLite has no native tz support; this keeps every datetime that leaves the
    DB comparable with ``datetime.now(timezone.utc)`` and rejects naive input
    so a local-time value can never be silently stored as UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; pass a tz-aware UTC datetime")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class RefundStatus(str, enum.Enum):
    NONE = "none"
    REFUNDED = "refunded"


class AuditAction(str, enum.Enum):
    REFUND_ISSUED = "refund_issued"
    REFUND_REJECTED_ALREADY_REFUNDED = "refund_rejected_already_refunded"
    REFUND_REJECTED_OUTSIDE_POLICY = "refund_rejected_outside_policy"
    REFUND_REJECTED_EMAIL_MISMATCH = "refund_rejected_email_mismatch"


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (CheckConstraint("amount > 0", name="ck_orders_amount_positive"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    purchase_date: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    refund_status: Mapped[RefundStatus] = mapped_column(
        Enum(RefundStatus, native_enum=False, validate_strings=True, length=16,
             values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=RefundStatus.NONE,
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False, index=True)
    action_taken: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction, native_enum=False, validate_strings=True, length=48,
             values_callable=lambda e: [m.value for m in e]), nullable=False
    )
    llm_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
