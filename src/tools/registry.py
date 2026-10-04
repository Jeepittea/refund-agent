"""Typed function-calling registry.

Each tool is a (Pydantic input model, Pydantic output model, handler) triple.
The registry is the *only* path from model output to handler code: raw tool
input is validated against the input model before the handler runs, and the
handler's return value is validated against the output model before it goes
back to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from src.database.session import Clock, utc_now
from src.tools.schemas import ToolError

log = logging.getLogger(__name__)

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


@dataclass(frozen=True)
class ToolContext:
    """Trusted, server-side facts a handler may rely on.

    ``sender_email`` comes from the parsed ``From:`` header of the inbound
    email, never from the model, so the model cannot act on another
    customer's order by putting a different address in a tool call.
    """

    session_factory: sessionmaker[Session]
    sender_email: str
    clock: Clock = utc_now


@dataclass(frozen=True)
class ToolSpec(Generic[InT, OutT]):
    name: str
    description: str
    input_model: type[InT]
    output_model: type[OutT]
    handler: Callable[[InT, ToolContext], OutT]

    def anthropic_definition(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        schema["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "input_schema": schema}


@dataclass(frozen=True)
class ToolResult:
    content: str  # JSON payload sent back to the model
    is_error: bool
    model: BaseModel  # validated output (or ToolError) for orchestrator bookkeeping


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def definitions(self) -> list[dict[str, Any]]:
        # Sorted for a deterministic tool list (keeps the prompt-cache prefix stable).
        return [self._tools[n].anthropic_definition() for n in sorted(self._tools)]

    def execute(self, name: str, raw_input: Any, ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return self._error(f"Unknown tool {name!r}")

        try:
            parsed = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            log.warning("Rejected invalid input for %s: %s", name, exc)
            return self._error(
                "Invalid tool input",
                [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()],
            )

        try:
            out = spec.handler(parsed, ctx)
            out = spec.output_model.model_validate(out)
        except Exception:  # noqa: BLE001 - never leak internals to the model
            log.exception("Tool %s failed", name)
            return self._error("Internal error while executing tool; no changes were made.")

        return ToolResult(content=out.model_dump_json(), is_error=False, model=out)

    @staticmethod
    def _error(message: str, details: list[dict] | None = None) -> ToolResult:
        err = ToolError(error=message, details=details)
        return ToolResult(content=err.model_dump_json(exclude_none=True), is_error=True, model=err)
