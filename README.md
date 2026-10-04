# Refund Agent

Autonomous customer-support refund agent built on Claude Opus 5.5 (`claude-opus-5-5`).

```
src/agent/      email parsing, agent loop, CLI
src/database/   SQLAlchemy models (orders, audit_log), engine, seed data
src/tools/      Pydantic tool schemas, typed tool registry, business-logic tools
tests/          pytest suite (offline, scripted fake Claude) + opt-in live tests
```

## Setup

```sh
uv venv -p 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv/Scripts/python -m src.agent.cli init-db          # creates + seeds refunds.db
.venv/Scripts/python -m pytest                          # offline suite
```

Process an email (needs `ANTHROPIC_API_KEY` or an `ant auth login` profile):

```sh
.venv/Scripts/python -m src.agent.cli handle samples/refund_request.eml
RUN_LIVE_TESTS=1 .venv/Scripts/python -m pytest -m live   # real-API smoke tests
```

## Who decides what

| Concern | Owner |
|---|---|
| Which tools to call, wording of the reply | LLM |
| Sender identity | Python: parsed `From:` header, bound into `ToolContext`; not a tool parameter |
| 30-day eligibility | Python `datetime` in `issue_refund` (`elapsed <= 30 days`, tz-aware UTC) |
| Final `Decision` | Python, derived from validated tool outputs; never from the model's text |

## Guarantees in `issue_refund`

* **One transaction.** `BEGIN IMMEDIATE` (SQLite) takes the write lock before the
  checks read the order. The idempotency check, policy check, status update and
  audit insert commit together or not at all.
* **Idempotent.** It checks status first, then does a compare-and-set
  `UPDATE ... WHERE refund_status='none'`. Duplicate emails, duplicate parallel
  tool calls and concurrent threads produce exactly one `refund_issued` row (tested
  with 8 racing threads).
* **Rejections are audited too.** Rejections are written to the audit log with the
  model's reasoning. Orders owned by another customer return `order_not_found`, so
  the tool can't be used to check whether an order exists.

Tool inputs are Pydantic models with `strict=True, extra="forbid"`: `"1001"` is
rejected for an int and unknown fields such as `within_refund_window` are rejected.
Validation errors go back to the model as `is_error` tool results.

Malformed emails (no headers, missing or invalid `From`, empty or oversized body,
structural defects) are rejected before any LLM call. A refusal, `max_tokens` stop,
API error or runaway loop returns `NEEDS_HUMAN_REVIEW`. The one exception: if a
refund has already been committed, the decision is reported as `REFUNDED` and no
reply is sent, so a human can write one.

Requests use adaptive thinking (Opus 5.5 default), `effort="medium"`, and
server-side refusal fallbacks (`fallbacks="default"`, beta
`server-side-fallback-2026-07-01`).
