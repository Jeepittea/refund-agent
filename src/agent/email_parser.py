"""Deterministic parsing/validation of raw inbound emails (RFC 5322).

This runs before the LLM. A malformed email never reaches the model or the
tools, and the sender address bound to tool calls comes from here.
"""

from __future__ import annotations

from email import message_from_string, policy
from email.message import EmailMessage
from email.utils import parseaddr

from pydantic import BaseModel, ConfigDict, EmailStr, Field, TypeAdapter, ValidationError

MAX_RAW_BYTES = 100_000
MAX_BODY_CHARS = 20_000

_email_adapter = TypeAdapter(EmailStr)


class MalformedEmailError(ValueError):
    pass


class InboundEmail(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sender: EmailStr
    subject: str = Field(max_length=998)
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)


def parse_raw_email(raw: str) -> InboundEmail:
    if not isinstance(raw, str) or not raw.strip():
        raise MalformedEmailError("Email is empty")
    if len(raw.encode("utf-8", errors="replace")) > MAX_RAW_BYTES:
        raise MalformedEmailError("Email exceeds maximum size")

    # A real message needs a header block; free text with no headers isn't one.
    head = raw.lstrip().split("\n", 1)[0]
    if ":" not in head:
        raise MalformedEmailError("Missing RFC 5322 header block")

    try:
        msg = message_from_string(raw, policy=policy.default)
    except Exception as exc:  # noqa: BLE001
        raise MalformedEmailError(f"Unparseable email: {exc}") from exc
    if msg.defects:
        raise MalformedEmailError(f"Email has structural defects: {msg.defects}")

    from_header = msg.get("From")
    if not from_header:
        raise MalformedEmailError("Missing From header")
    _, addr = parseaddr(str(from_header))
    try:
        sender = _email_adapter.validate_python(addr)
    except ValidationError as exc:
        raise MalformedEmailError(f"Invalid From address: {addr!r}") from exc

    body = _extract_text_body(msg).strip()
    if not body:
        raise MalformedEmailError("Email has no text body")

    try:
        return InboundEmail(
            sender=sender.lower(),
            subject=str(msg.get("Subject", "")).strip(),
            body=body,
        )
    except ValidationError as exc:
        raise MalformedEmailError(f"Email failed validation: {exc}") from exc


def _extract_text_body(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain",))
    if part is None:
        return ""
    try:
        return part.get_content()
    except (LookupError, ValueError):
        return ""
