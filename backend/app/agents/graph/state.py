"""State shared by every node of the agent graph.

Most fields are rebuilt each turn; only RESUME_STATE_FIELDS are saved while a turn waits on a
clarification question.
"""

from typing import Literal, NotRequired, TypedDict

Intent = Literal["qa_agent", "study_tools_agent", "summary_agent", "study_planner_agent"]
ContextStatus = Literal["empty_course", "not_ready", "ok"]


class NeedsClarification(TypedDict):
    pending_agent: Intent | None  # None when the orchestrator asks on its own initiative
    reason: str
    suggested_question: str


class AgentResult(TypedDict):
    agent: Intent
    # e.g. "quiz", "flashcards", "summary"; the same (agent, kind) never runs twice per request.
    kind: str
    summary: str  # short, student-facing confirmation; compose_response joins/synthesizes these
    content: str  # the fuller material this step produced; source for a dependent next delegation
    # Q&A only: questions the student might ask next, offered as chips under the reply.
    follow_ups: NotRequired[list[str]]


class AgentState(TypedDict):
    # Rebuilt every turn.
    messages: list[tuple[str, str]]  # (role, content) conversation history, oldest first
    course_id: int | None  # None for a global study-planner session
    session_id: int
    raw_message: str  # the student's latest message, unmodified

    # What the orchestrator delegated this turn and what came back; reset every turn.
    scratchpad: list[tuple[str, str]]
    orchestrator_steps: int  # loop guard for the orchestrator's own delegate/ask/finish loop

    # Set by the orchestrator when it delegates.
    pending_delegation: Intent | None
    retrieval_query: str
    delegation_focus: str  # which part of the student's message this delegation is for
    use_previous_result: bool  # use the previous result's content instead of a fresh search
    # AgentResult.kind for this delegation, e.g. "summary", "quiz", "answer:<focus>", "plan".
    delegation_kind: str
    done: bool  # the orchestrator decided to stop delegating

    context_status: ContextStatus
    retrieved_context: str

    # The clarification this turn answers, or None on a fresh request.
    resumed_from: NeedsClarification | None

    # Resume state: saved to chat_sessions.pending_state while waiting on a clarification.
    needs_clarification: NeedsClarification | None
    # Every result for the current request, including those from before a clarification.
    agent_results: list[AgentResult]
    # How many agent_results the student has already been told about.
    reported_count: int
    # The message that started the current request (differs from raw_message when resuming).
    request_message: str

    # Set by the two student-facing terminal nodes (ask_clarification, compose_response).
    final_message: str | None
    handled_by: list[str]  # the agents whose results this turn's reply reports
    follow_ups: list[str]  # suggested next questions for a finished turn (compose_response)


RESUME_STATE_FIELDS = (
    "needs_clarification",
    "agent_results",
    "reported_count",
    "request_message",
)
