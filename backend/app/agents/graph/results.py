"""Helpers that build a sub-agent's state update (a success or a clarification request)."""

from app.agents.graph.state import AgentResult, AgentState, Intent, NeedsClarification


def success(
    state: AgentState,
    agent: Intent,
    message: str,
    *,
    content: str | None = None,
    kind: str | None = None,
    follow_ups: list[str] | None = None,
) -> dict:
    """`kind` defaults to the orchestrator's `delegation_kind`."""
    entry: AgentResult = {
        "agent": agent,
        "kind": kind or state.get("delegation_kind") or agent,
        "summary": message,
        "content": content if content is not None else message,
    }
    if follow_ups:
        entry["follow_ups"] = follow_ups
    return {"agent_results": [*state["agent_results"], entry], "needs_clarification": None}


def clarify(agent: Intent, reason: str, suggested_question: str) -> dict:
    pending: NeedsClarification = {
        "pending_agent": agent,
        "reason": reason,
        "suggested_question": suggested_question,
    }
    return {"needs_clarification": pending}


def task_message(state: AgentState) -> str:
    """The task for a specialist: its part of the message (if focused) plus the latest message."""
    focus = state["delegation_focus"]
    if not focus:
        return state["raw_message"]
    return (
        f"Your task (your part of the student's request): {focus}\n\n"
        f"The student's latest message, for any details it gives: {state['raw_message']}"
    )
