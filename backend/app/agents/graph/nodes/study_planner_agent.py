"""Study planner node: plans study sessions around the student's Google Calendar.

Runs its own tool loop with two kinds of tools: calendar tools discovered from the Calendar MCP
server, and app tools defined here (remember events/preferences, save and confirm a plan).
"""

import json
import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import llm
from app.agents.graph import results
from app.agents.graph.state import AgentState
from app.auth.tokens import get_valid_google_access_token
from app.mcp import calendar
from app.models import CalendarEventResolution, Course, CourseStatus, Material, User
from app.planner import memory, plans

logger = logging.getLogger(__name__)

AGENT_NAME = "study_planner_agent"
MAX_STEPS = 14
CALENDAR_DAYS = 14  # how far ahead the prompt lists named days

CALENDAR_UNAVAILABLE = (
    "I couldn't reach your Google Calendar just now, so I can't build a plan. Please try again "
    "in a moment."
)
NO_COURSES_MESSAGE = (
    "You don't have any active courses yet. Create a course and upload its materials, and I can "
    "plan study sessions around your exams and deadlines."
)
FALLBACK_MESSAGE = (
    "I couldn't put a plan together. Could you tell me a bit more about what you need?"
)

ASK_TOOL = "ask_student"
FINISH_TOOL = "finish"
REMEMBER_EVENT_TOOL = "remember_event"
REMEMBER_PREFERENCES_TOOL = "remember_preferences"
SAVE_PLAN_TOOL = "save_plan"
CONFIRM_PLAN_TOOL = "confirm_plan"

APP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": ASK_TOOL,
            "description": (
                "Ask the student one question and pause. Use this when you cannot proceed without "
                "their answer - which course an exam belongs to, or which materials it covers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you need this, for the log. Not shown to the student.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to put to the student, in Markdown.",
                    },
                },
                "required": ["reason", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINISH_TOOL,
            "description": "Finish this turn and report back to the student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "What to tell the student, in Markdown. If you built a plan, lay the "
                            "sessions out clearly. If your schedule was too tight to fit "
                            "everything, say so plainly rather than silently dropping sessions."
                        ),
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": REMEMBER_EVENT_TOOL,
            "description": (
                "Record what a calendar exam or deadline refers to, once the student has told "
                "you. Remembered for good, so you never ask about this same event again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "google_event_id": {"type": "string"},
                    "event_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "event_title": {"type": "string"},
                    "course_id": {
                        "type": "integer",
                        "description": "The course this event belongs to; omit if still unknown.",
                    },
                    "materials_covered": {
                        "type": "string",
                        "description": "What the student said it covers, in their own words.",
                    },
                },
                "required": ["google_event_id", "event_date", "event_title", "materials_covered"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": REMEMBER_PREFERENCES_TOOL,
            "description": (
                "Store or update the student's study preferences. Pass only what they told you; "
                "anything omitted keeps its current value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_study_time": {
                        "type": "string",
                        "description": 'e.g. "morning", "evening", "night"',
                    },
                    "sleep_start": {"type": "string", "description": "HH:MM"},
                    "sleep_end": {"type": "string", "description": "HH:MM"},
                    "time_zone": {"type": "string", "description": 'IANA zone, e.g. "Asia/Beirut"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SAVE_PLAN_TOOL,
            "description": (
                "Save the study sessions you've worked out as a draft plan. This does not touch "
                "the calendar. Returns a plan_id to confirm later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_range_start": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_range_end": {"type": "string", "description": "YYYY-MM-DD"},
                    "sessions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "course_id": {"type": "integer"},
                                "start_time": {
                                    "type": "string",
                                    "description": "RFC3339 with offset",
                                },
                                "end_time": {
                                    "type": "string",
                                    "description": "RFC3339 with offset",
                                },
                                "topic": {
                                    "type": "string",
                                    "description": 'What to study, e.g. "Chapter 3 - Integrals"',
                                },
                            },
                            "required": ["course_id", "start_time", "end_time", "topic"],
                        },
                    },
                    "plan_id": {
                        "type": "integer",
                        "description": "Pass the existing plan_id when revising a draft.",
                    },
                },
                "required": ["date_range_start", "date_range_end", "sessions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CONFIRM_PLAN_TOOL,
            "description": (
                "Write a saved plan's sessions to the student's Google Calendar. Only call this "
                "once the student has agreed to the plan, explicitly or by moving on without "
                "objecting. Calling it again on the same plan updates the existing events rather "
                "than creating duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "integer"}},
                "required": ["plan_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are the study planner for a student, working across all of their active courses at once.

Right now it is {now}.
{zone_note}
The coming days, for naming them - always take weekday names from this list, never work them out \
yourself:
{days}
"Today" and "tomorrow" are the first two entries above. "The next N days" means today and the \
following N-1 days. Events earlier today than the current time have already happened.

Their active courses (use these course_id values):
{courses}

Stored preferences: {preferences}

Events they have already told you about - never ask about these again:
{resolutions}

How to work:
1. Read their calendar for the planning window with the calendar tools. Exams and deadlines there \
are theirs: read them, never create or change them.
2. Work out which course each exam or deadline belongs to. If the title makes it obvious, just \
proceed. If it's genuinely ambiguous, ask.
3. Always ask which materials or topics an exam covers, unless it's in the list above. Never guess \
this from the event title alone.
4. Ask about preferences once if none are stored, then remember them.
5. Plan within the window the student asked for. If they didn't give one, ask for it - never plan \
open-endedly.
6. Prioritise the soonest deadline first, and exams over non-exam deadlines at similar urgency.
7. Schedule only in genuinely free time, around what's already on the calendar and outside their \
sleeping hours.
8. Always produce the best plan the week allows. If there isn't enough free time, still plan what \
fits and say plainly that it's tight - never refuse to plan.
9. Save the plan, show it to the student, and confirm it only once they've agreed.

Ask one question at a time with {ask_tool}, and use {finish_tool} when you're done.\
"""


async def study_planner_agent(state: AgentState, *, db: AsyncSession, user: User) -> dict:
    """One planner turn: a ReAct loop over the MCP calendar tools plus this node's own tools."""
    courses = await _active_courses(db, user.id)
    if not courses:
        return results.success(state, AGENT_NAME, NO_COURSES_MESSAGE)

    try:
        access_token = await get_valid_google_access_token(db, user)
    except Exception as exc:  # noqa: BLE001 - no Google token, or refresh refused
        logger.warning("Planner: no usable Google token for user %s: %s", user.id, exc)
        return results.success(state, AGENT_NAME, CALENDAR_UNAVAILABLE)

    try:
        async with calendar.session() as mcp:
            tools = [tool.as_openai_tool() for tool in await mcp.list_tools()] + APP_TOOLS
            return await _run_loop(state, db, user, courses, mcp, access_token, tools)
    except calendar.CalendarMcpError as exc:
        logger.warning("Planner: calendar MCP unavailable: %s", exc)
        return results.success(state, AGENT_NAME, CALENDAR_UNAVAILABLE)


async def _run_loop(
    state: AgentState,
    db: AsyncSession,
    user: User,
    courses: list[Course],
    mcp: calendar.CalendarMcpSession,
    access_token: str,
    tools: list[dict[str, Any]],
) -> dict:
    preferences = await memory.get_preferences(db, user.id)
    system = SYSTEM_PROMPT.format(
        **_time_context(preferences.time_zone if preferences else None),
        courses=await _describe_courses(db, courses),
        preferences=memory.describe_preferences(preferences),
        resolutions=await _describe_resolutions(db, user.id),
        ask_tool=ASK_TOOL,
        finish_tool=FINISH_TOOL,
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages += [{"role": role, "content": content} for role, content in state["messages"]]
    messages.append({"role": "user", "content": state["raw_message"]})

    for step in range(MAX_STEPS):
        reply = await llm.complete(messages, tools=tools)
        if not reply.tool_calls:
            # No tool call means the model wrote prose instead of finishing properly; take it.
            return results.success(state, AGENT_NAME, reply.content or FALLBACK_MESSAGE)

        messages.append(_assistant_turn(reply))

        for call in reply.tool_calls:
            if call.name == ASK_TOOL:
                question = call.arguments.get("question") or FALLBACK_MESSAGE
                return results.clarify(
                    AGENT_NAME, call.arguments.get("reason", "needs input"), question
                )
            if call.name == FINISH_TOOL:
                return results.success(
                    state, AGENT_NAME, call.arguments.get("message") or FALLBACK_MESSAGE
                )

            output = await _run_tool(db, user, mcp, access_token, call, preferences)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _as_text(output)})

        logger.debug("Planner step %s for user %s", step + 1, user.id)

    logger.warning("Planner: hit the %s-step limit for user %s", MAX_STEPS, user.id)
    return results.success(state, AGENT_NAME, FALLBACK_MESSAGE)


async def _run_tool(
    db: AsyncSession,
    user: User,
    mcp: calendar.CalendarMcpSession,
    access_token: str,
    call: llm.ToolCall,
    preferences: Any,
) -> Any:
    """Dispatch one tool call: this node's own tools first, anything else over MCP."""
    arguments = call.arguments

    if call.name == REMEMBER_EVENT_TOOL:
        try:
            event_date = date.fromisoformat(arguments["event_date"])
        except (KeyError, ValueError):
            return {"error": "event_date must be YYYY-MM-DD."}
        await memory.save_resolution(
            db,
            user.id,
            google_event_id=arguments.get("google_event_id", ""),
            event_date=event_date,
            event_title=arguments.get("event_title", ""),
            course_id=arguments.get("course_id"),
            materials_covered=arguments.get("materials_covered", ""),
        )
        return {"remembered": True}

    if call.name == REMEMBER_PREFERENCES_TOOL:
        await memory.save_preferences(
            db,
            user.id,
            preferred_study_time=arguments.get("preferred_study_time"),
            sleep_start=arguments.get("sleep_start"),
            sleep_end=arguments.get("sleep_end"),
            time_zone=arguments.get("time_zone"),
        )
        return {"saved": True}

    if call.name == SAVE_PLAN_TOOL:
        return await _save_plan(db, user, arguments)

    if call.name == CONFIRM_PLAN_TOOL:
        return await _confirm_plan(db, user, mcp, access_token, arguments, preferences)

    # Anything else is a calendar tool: it goes over MCP, with the token injected there.
    outcome = await mcp.call_tool(call.name, arguments, access_token=access_token)
    if outcome.auth_rejected:
        # The backend owns the token, so refresh once and let the model try again.
        logger.info("Planner: calendar rejected the token, refreshing for user %s", user.id)
        refreshed = await get_valid_google_access_token(db, user)
        outcome = await mcp.call_tool(call.name, arguments, access_token=refreshed)
    if outcome.is_error:
        return {"error": outcome.content}
    return outcome.content


async def _save_plan(db: AsyncSession, user: User, arguments: dict[str, Any]) -> Any:
    try:
        start = date.fromisoformat(arguments["date_range_start"])
        end = date.fromisoformat(arguments["date_range_end"])
    except (KeyError, ValueError):
        return {"error": "date_range_start and date_range_end must be YYYY-MM-DD."}

    sessions = []
    for raw in arguments.get("sessions", []):
        try:
            sessions.append(
                plans.DraftSession(
                    course_id=int(raw["course_id"]),
                    start_time=_parse_timestamp(raw["start_time"]),
                    end_time=_parse_timestamp(raw["end_time"]),
                    topic=str(raw["topic"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": f"Bad session {raw!r}: {exc}"}
    if not sessions:
        return {"error": "A plan needs at least one session."}

    plan = await plans.save_draft(
        db,
        user.id,
        date_range_start=start,
        date_range_end=end,
        sessions=sessions,
        plan_id=arguments.get("plan_id"),
    )
    return {"plan_id": plan.id, "sessions_saved": len(sessions)}


async def _confirm_plan(
    db: AsyncSession,
    user: User,
    mcp: calendar.CalendarMcpSession,
    access_token: str,
    arguments: dict[str, Any],
    preferences: Any,
) -> Any:
    plan_id = arguments.get("plan_id")
    plan = None
    if plan_id is not None:
        from app.models import StudyPlan

        plan = await db.scalar(
            select(StudyPlan).where(StudyPlan.id == plan_id, StudyPlan.user_id == user.id)
        )
    if plan is None:
        plan = await plans.latest_plan(db, user.id)
    if plan is None:
        return {"error": "No plan to confirm - save one first."}

    result = await plans.confirm_plan(
        db,
        plan=plan,
        mcp=mcp,
        access_token=access_token,
        time_zone=getattr(preferences, "time_zone", None),
    )
    return {
        "plan_id": plan.id,
        "events_created": result.written,
        "events_updated": result.updated,
        "failures": result.failures,
    }


async def _active_courses(db: AsyncSession, user_id: int) -> list[Course]:
    rows = await db.scalars(
        select(Course)
        .where(Course.user_id == user_id, Course.status == CourseStatus.active)
        .order_by(Course.name)
    )
    return list(rows)


async def _describe_courses(db: AsyncSession, courses: list[Course]) -> str:
    """Course list with material titles, so the planner can talk about topics concretely."""
    lines = []
    for course in courses:
        titles = await db.scalars(
            select(Material.title).where(Material.course_id == course.id).limit(25)
        )
        materials = ", ".join(titles) or "no materials uploaded"
        lines.append(
            f"- id={course.id} {course.name} ({course.semester} {course.year}): {materials}"
        )
    return "\n".join(lines)


async def _describe_resolutions(db: AsyncSession, user_id: int) -> str:
    rows = await db.scalars(
        select(CalendarEventResolution)
        .where(CalendarEventResolution.user_id == user_id)
        .order_by(CalendarEventResolution.event_date.desc())
        .limit(40)
    )
    lines = [
        f"- {row.event_date.isoformat()} {row.event_title!r} (event_id={row.google_event_id}): "
        f"course_id={row.course_id}, covers: {row.materials_covered}"
        for row in rows
    ]
    return "\n".join(lines) if lines else "(none yet)"


def _assistant_turn(reply: llm.LlmReply) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in reply.tool_calls
        ],
    }


def _now() -> datetime:
    """The current moment; a function so tests can fix it."""
    return datetime.now(UTC)


def _time_context(time_zone: str | None, now: datetime | None = None) -> dict[str, str]:
    """Current time and upcoming day names for the prompt, in the student's time zone (or UTC)."""
    zone_name = time_zone or "UTC"
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):  # an unknown or malformed zone name
        zone, zone_name = ZoneInfo("UTC"), "UTC"
    local = (now or _now()).astimezone(zone)
    days = (local.date() + timedelta(days=offset) for offset in range(CALENDAR_DAYS))
    known = time_zone is not None and zone_name == time_zone
    # local.day rather than %-d, which Windows' strftime doesn't support.
    return {
        "now": f"{local:%A} {local.day} {local:%B %Y}, {local:%H:%M} ({zone_name})",
        "zone_note": (
            ""
            if known
            else "The student's time zone isn't known yet, so this is UTC; ask for it and "
            "remember it with the preferences tool before relying on the local date."
        ),
        "days": " · ".join(f"{day:%a} {day.day} {day:%b}" for day in days),
    }


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    # Treat a timestamp without an offset as UTC.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
