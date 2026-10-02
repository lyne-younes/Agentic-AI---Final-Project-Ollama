"""Draft study plans and confirming them to Google Calendar.

`confirm_plan` is the only function that writes study sessions to the calendar. A session that
already has an event id is updated instead of created again.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.calendar import CalendarMcpSession
from app.models import Course, StudyPlan, StudyPlanSession, StudyPlanStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftSession:
    """One proposed study block, before it exists on the calendar."""

    course_id: int
    start_time: datetime
    end_time: datetime
    topic: str


@dataclass(frozen=True)
class ConfirmationResult:
    written: int
    updated: int
    failures: list[str]


async def save_draft(
    db: AsyncSession,
    user_id: int,
    *,
    date_range_start: date,
    date_range_end: date,
    sessions: list[DraftSession],
    plan_id: int | None = None,
) -> StudyPlan:
    """Create or overwrite a draft plan (its sessions are replaced each time)."""
    plan = None
    if plan_id is not None:
        plan = await db.scalar(
            select(StudyPlan).where(StudyPlan.id == plan_id, StudyPlan.user_id == user_id)
        )
    if plan is None:
        plan = StudyPlan(user_id=user_id)
        db.add(plan)

    if plan.status == StudyPlanStatus.confirmed:
        # A confirmed plan isn't changed; start a new draft instead.
        plan = StudyPlan(user_id=user_id)
        db.add(plan)

    plan.date_range_start = date_range_start
    plan.date_range_end = date_range_end
    plan.status = StudyPlanStatus.draft
    await db.flush()

    await db.execute(delete(StudyPlanSession).where(StudyPlanSession.study_plan_id == plan.id))
    db.add_all(
        [
            StudyPlanSession(
                study_plan_id=plan.id,
                course_id=session.course_id,
                start_time=session.start_time,
                end_time=session.end_time,
                topic=session.topic,
            )
            for session in sessions
        ]
    )
    await db.flush()
    return plan


async def sessions_for(db: AsyncSession, plan_id: int) -> list[StudyPlanSession]:
    rows = await db.scalars(
        select(StudyPlanSession)
        .where(StudyPlanSession.study_plan_id == plan_id)
        .order_by(StudyPlanSession.start_time)
    )
    return list(rows)


async def describe_plan(db: AsyncSession, plan: StudyPlan) -> str:
    """A compact, student-readable rendering of a plan, for the agent to quote back."""
    sessions = await sessions_for(db, plan.id)
    if not sessions:
        return "(no sessions)"
    names = await _course_names(db, {session.course_id for session in sessions})
    lines = []
    for session in sessions:
        course = names.get(session.course_id, "Unknown course")
        lines.append(
            f"- {session.start_time:%a %d %b %H:%M}-{session.end_time:%H:%M} "
            f"{course}: {session.topic}"
        )
    return "\n".join(lines)


async def _course_names(db: AsyncSession, course_ids: set[int]) -> dict[int, str]:
    if not course_ids:
        return {}
    rows = await db.execute(select(Course.id, Course.name).where(Course.id.in_(course_ids)))
    return {course_id: name for course_id, name in rows.all()}


async def confirm_plan(
    db: AsyncSession,
    *,
    plan: StudyPlan,
    mcp: CalendarMcpSession,
    access_token: str,
    time_zone: str | None = None,
) -> ConfirmationResult:
    """Write a plan's sessions to Google Calendar through MCP and record the event ids."""
    sessions = await sessions_for(db, plan.id)
    names = await _course_names(db, {session.course_id for session in sessions})

    written = 0
    updated = 0
    failures: list[str] = []

    for session in sessions:
        course_name = names.get(session.course_id, "Study")
        title = f"Study: {session.topic} - {course_name}"
        arguments = {
            "title": title,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat(),
            "description": f"{course_name}: {session.topic}",
        }
        if time_zone:
            arguments["time_zone"] = time_zone

        if session.google_calendar_event_id:
            outcome = await mcp.call_tool(
                "update_study_session",
                {**arguments, "event_id": session.google_calendar_event_id},
                access_token=access_token,
            )
            if not outcome.is_error:
                updated += 1
        else:
            outcome = await mcp.call_tool(
                "create_study_session", arguments, access_token=access_token
            )
            if not outcome.is_error and isinstance(outcome.content, dict):
                session.google_calendar_event_id = outcome.content.get("event_id")
                written += 1

        if outcome.is_error:
            logger.warning("confirm_plan: %s failed: %s", title, outcome.content)
            failures.append(f"{title}: {outcome.content}")

    if written or updated:
        plan.status = StudyPlanStatus.confirmed
        plan.confirmed_at = datetime.now(UTC)
    # Commit now: the events already exist in Google, so a later rollback must not lose their ids.
    await db.commit()
    return ConfirmationResult(written=written, updated=updated, failures=failures)


async def latest_plan(db: AsyncSession, user_id: int) -> StudyPlan | None:
    """The plan a bare "confirm" refers to: the student's most recent one."""
    return await db.scalar(
        select(StudyPlan).where(StudyPlan.user_id == user_id).order_by(StudyPlan.id.desc()).limit(1)
    )
