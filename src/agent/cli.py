"""Command-line entry point.

    python -m src.agent.cli init-db [--db refunds.db]
    python -m src.agent.cli handle path/to/email.eml [--db refunds.db]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agent.orchestrator import RefundAgent
from src.database import init_db, make_engine, make_session_factory, seed_mock_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="refund-agent")
    parser.add_argument("--db", default="refunds.db", help="SQLite file path")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="Create tables and seed mock orders")
    handle = sub.add_parser("handle", help="Process one raw email file")
    handle.add_argument("email_file", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = make_engine(f"sqlite:///{args.db}")
    init_db(engine)
    sessions = make_session_factory(engine)

    if args.cmd == "init-db":
        seeded = seed_mock_data(sessions)
        print(f"Database ready at {args.db}; seeded {len(seeded)} orders.")
        return 0

    result = RefundAgent(sessions).handle_email(args.email_file.read_text(encoding="utf-8"))
    print(json.dumps({
        "decision": result.decision.value,
        "sender": result.sender,
        "detail": result.detail,
        "tool_calls": [
            {"name": c.name, "input": c.input, "is_error": c.is_error,
             "output": c.output.model_dump(mode="json")}
            for c in result.tool_calls
        ],
    }, indent=2))
    if result.reply_email:
        print("\n--- reply ---\n" + result.reply_email)
    return 0


if __name__ == "__main__":
    sys.exit(main())
