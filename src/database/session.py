"""Engine/session factory, schema creation and mock-data seeding."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, Order, RefundStatus

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_engine(url: str = "sqlite:///refunds.db", echo: bool = False) -> Engine:
    """Create an engine with correct SQLite transaction semantics.

    pysqlite's default driver behaviour defers BEGIN until the first write and
    therefore does not give us a serializable read-check-write. Per the
    SQLAlchemy docs we take over transaction control and emit
    ``BEGIN IMMEDIATE`` so the write lock is acquired *before* the idempotency
    check reads the order row. Two concurrent refunds for the same order are
    therefore strictly serialized.
    """
    engine = create_engine(url, echo=echo, connect_args={"timeout": 30})

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, _record):
            dbapi_conn.isolation_level = None  # disable pysqlite's implicit BEGIN
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn):
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def seed_mock_data(session_factory: sessionmaker[Session], clock: Clock = utc_now) -> list[Order]:
    """Insert deterministic mock orders relative to ``clock()``. No-op if orders exist."""
    now = clock()
    orders = [
        Order(id=1001, customer_email="alice@example.com", amount=Decimal("49.99"),
              purchase_date=now - timedelta(days=5), refund_status=RefundStatus.NONE),
        Order(id=1002, customer_email="bob@example.com", amount=Decimal("129.00"),
              purchase_date=now - timedelta(days=45), refund_status=RefundStatus.NONE),
        Order(id=1003, customer_email="carol@example.com", amount=Decimal("15.50"),
              purchase_date=now - timedelta(days=10), refund_status=RefundStatus.REFUNDED),
        Order(id=1004, customer_email="dave@example.com", amount=Decimal("300.00"),
              purchase_date=now - timedelta(days=29, hours=23), refund_status=RefundStatus.NONE),
        Order(id=1005, customer_email="erin@example.com", amount=Decimal("75.25"),
              purchase_date=now - timedelta(days=30, minutes=1), refund_status=RefundStatus.NONE),
    ]
    with session_factory.begin() as session:
        if session.scalar(select(Order.id).limit(1)) is not None:
            return []
        session.add_all(orders)
    return orders
