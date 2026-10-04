# refund-agent
AI customer-support agent that handles refund emails using Claude Opus 5.5 and tool calling. Refund rules are enforced in Python, not by the AI: a strict 30-day policy check, no double refunds, and all-or-nothing database saves with an audit log. Built with SQLAlchemy, Pydantic and Pytest.
