"""Bookkeeping node plus the two nodes that write the reply to the student."""

from app import llm
from app.agents.graph.state import AgentResult, AgentState

FALLBACK_MESSAGE = "I wasn't able to figure out how to help with that. Could you rephrase?"

SYNTHESIS_SYSTEM_PROMPT = """\
You are combining results from several specialist assistants into one reply to a student, in a \
single natural voice. The student wrote: "{request}"

Cover every result below exactly once - don't drop any, don't repeat any, and don't claim \
anything they don't say - but merge them into one coherent message rather than pasting them \
back to back.

Results:
{results}\
"""


async def record_result(state: AgentState) -> dict:
    """Notes the specialist's outcome in the orchestrator's scratchpad."""
    if state["needs_clarification"]:
        c = state["needs_clarification"]
        note = (
            f"[{c['pending_agent']} needs clarification] {c['reason']}: {c['suggested_question']}"
        )
    else:
        last = state["agent_results"][-1]
        note = f"[{last['agent']} result] {last['summary']}"
    return {"scratchpad": [*state["scratchpad"], ("user", note)]}


async def ask_clarification(state: AgentState) -> dict:
    """Asks the clarification question, after reporting any results finished so far."""
    clarification = state["needs_clarification"]
    assert clarification is not None  # route_after_orchestrator only sends us here when it's set
    new = state["agent_results"][state["reported_count"] :]
    parts = [result["summary"] for result in new] + [clarification["suggested_question"]]
    agents = [result["agent"] for result in new] + [
        clarification["pending_agent"] or "orchestrator"
    ]
    return {
        "final_message": "\n\n".join(parts),
        "handled_by": _unique(agents),
        "reported_count": len(state["agent_results"]),
    }


async def compose_response(state: AgentState) -> dict:
    """Combines this turn's new results into one reply."""
    results = state["agent_results"][state["reported_count"] :]
    handled_by = _unique([result["agent"] for result in results])
    # Suggested next questions come from this turn's latest answer that has any (Q&A only).
    follow_ups = next(
        (result["follow_ups"] for result in reversed(results) if result.get("follow_ups")), []
    )
    if not results:
        return {"final_message": FALLBACK_MESSAGE, "handled_by": handled_by, "follow_ups": []}
    if len(results) == 1:
        return {
            "final_message": results[0]["summary"],
            "handled_by": handled_by,
            "follow_ups": follow_ups,
        }
    message = await _synthesize(state["raw_message"], results)
    return {"final_message": message, "handled_by": handled_by, "follow_ups": follow_ups}


async def _synthesize(request: str, results: list[AgentResult]) -> str:
    listed = "\n".join(f"- {result['agent']}: {result['summary']}" for result in results)
    prompt = SYNTHESIS_SYSTEM_PROMPT.format(request=request, results=listed)
    # Some chat templates reject a request with only a system message, so it gets a user turn.
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Write the combined reply to the student."},
    ]
    reply = await llm.complete(messages)
    text = (reply.content or "").strip()
    if not text or reply.truncated:
        return "\n\n".join(result["summary"] for result in results)
    return text


def _unique(agents: list[str]) -> list[str]:
    return list(dict.fromkeys(agents))
