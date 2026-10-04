from __future__ import annotations

import itertools
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from src.database import init_db, make_engine, make_session_factory, seed_mock_data

FIXED_NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)


def fixed_clock() -> datetime:
    return FIXED_NOW


@pytest.fixture
def engine(tmp_path):
    # File-backed (not :memory:) so locking and multi-connection behaviour are real.
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def sessions(engine):
    factory = make_session_factory(engine)
    seed_mock_data(factory, clock=fixed_clock)
    return factory


# ------------------------------------------------------------ fake Claude API

_ids = itertools.count(1)


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, input: dict):
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids):04d}", name=name, input=input)


def response(*content, stop_reason: str | None = None, stop_details=None):
    if stop_reason is None:
        stop_reason = "tool_use" if any(c.type == "tool_use" for c in content) else "end_turn"
    return SimpleNamespace(content=list(content), stop_reason=stop_reason,
                           stop_details=stop_details)


class FakeClaude:
    """Stands in for ``anthropic.Anthropic``; replays scripted responses and records requests."""

    def __init__(self, *scripted):
        self._script = list(scripted)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot messages: the orchestrator mutates the list after the call returns.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._script:
            raise AssertionError("FakeClaude ran out of scripted responses")
        return self._script.pop(0)

    def tool_results(self, request_index: int) -> list[dict]:
        """tool_result blocks the orchestrator sent in the given request."""
        last = self.requests[request_index]["messages"][-1]
        return [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


def raw_email(sender: str, body: str, subject: str = "Refund request") -> str:
    return f"From: Customer <{sender}>\nTo: support@shop.example\nSubject: {subject}\n\n{body}\n"
