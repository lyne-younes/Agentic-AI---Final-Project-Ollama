"""Study tools node: creates one quiz or one flashcard set per delegation.

Two model calls: `resolve_request` works out what is being asked for (and may ask the student for
missing details), then a second call generates the content.
"""

import logging
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import llm
from app.agents.graph import results
from app.agents.graph.nodes import qa_agent
from app.agents.graph.state import AgentState
from app.models import (
    Course,
    Flashcard,
    FlashcardSet,
    QuestionType,
    Quiz,
    QuizDifficulty,
    QuizFormat,
    QuizQuestion,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "study_tools_agent"
DEFAULT_QUIZ_COUNT = 10
DEFAULT_FLASHCARD_COUNT = 15
QUIZ_FORMATS = {f.value for f in QuizFormat}
QUIZ_DIFFICULTIES = {d.value for d in QuizDifficulty}
QUESTION_TYPES = {t.value for t in QuestionType}

# How a student spells out a quiz format, for `_fill_stated_quiz_details`.
STATED_FORMATS = (
    ("mcq", r"\bmcqs?\b|multiple[\s-]*choice"),
    ("true_false", r"\btrue\s*(?:/|or|-|and)?\s*false\b|\bt/f\b"),
    ("short_answer", r"short[\s-]*answers?"),
    ("mixed", r"\bmixed\b"),
)

GENERATION_FAILED_MESSAGE = "I wasn't able to generate that just now. Please try again."

RESOLVE_TOOL_NAME = "resolve_request"
RESOLVE_TOOL = {
    "type": "function",
    "function": {
        "name": RESOLVE_TOOL_NAME,
        "description": "Decide what study material to generate, before generating it.",
        "parameters": {
            "type": "object",
            "properties": {
                "study_tool_type": {
                    "type": "string",
                    "enum": ["quiz", "flashcards", "unclear"],
                    "description": (
                        "unclear only if the request genuinely doesn't say whether the student "
                        "wants a quiz or flashcards (e.g. 'help me study chapter 3')."
                    ),
                },
                "covered": {
                    "type": "boolean",
                    "description": (
                        "True if the retrieved passages give enough to generate from and the "
                        "topic is on-topic for this course. False if the requested topic isn't "
                        "covered by any uploaded material - study material must never be "
                        "generated on content the student never uploaded."
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": sorted(QUIZ_FORMATS),
                    "description": "Quiz only. Omit if the student didn't specify one.",
                },
                "difficulty": {
                    "type": "string",
                    "enum": sorted(QUIZ_DIFFICULTIES),
                    "description": "Quiz only. Omit if the student didn't specify one.",
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "Only when the student gives an explicit number ('5 questions', '20 "
                        "cards'). 'a quiz', 'a flashcard' or 'some flashcards' is not a number - "
                        "omit it."
                    ),
                },
            },
            "required": ["study_tool_type", "covered"],
        },
    },
}

RESOLVE_SYSTEM_PROMPT = """\
You generate study material (quizzes and flashcards) for a student's course.

Course: {name} ({semester} {year})
Description: {description}

Passages retrieved from the course's uploaded materials for this request:
{context}

Read the conversation and call {tool} to say what to do next. If you've been given a task \
that is one part of the student's message, resolve that task only - other parts are handled \
separately. Format and difficulty are only relevant for a quiz; leave them out for flashcards or \
when unspecified.{assigned}\
"""

# Added when the orchestrator already chose the item type.
ASSIGNED_ITEM = """

You have been asked to make {article} {item}: set study_tool_type to {item}, and take its details \
(covered, count{quiz_details}) from anything the conversation says about the {item}."""

SUBMIT_QUIZ_TOOL_NAME = "submit_quiz"
SUBMIT_QUIZ_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_QUIZ_TOOL_NAME,
        "description": "Submit the generated quiz.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "question_type": {"type": "string", "enum": sorted(QUESTION_TYPES)},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "MCQ choices only; omit for other question types.",
                            },
                            "correct_answer": {
                                "type": "string",
                                "description": (
                                    "For MCQ, exactly one of options. For true/false, 'true' or "
                                    "'false'. For short answer, the expected answer text."
                                ),
                            },
                        },
                        "required": ["question_text", "question_type", "correct_answer"],
                    },
                },
            },
            "required": ["title", "questions"],
        },
    },
}

GENERATE_QUIZ_SYSTEM_PROMPT = """\
Generate a {difficulty}-difficulty, {format}-format quiz for the course "{name}", with up to \
{count} questions - fewer only if the material below doesn't support that many good questions; \
never pad with filler. Ground every question in the passages below; do not invent facts.

Passages:
{context}
{previous}
Call {tool} with the finished quiz.\
"""

# Lists earlier questions/card fronts so a new item doesn't repeat them.
PREVIOUS_ITEMS_PROMPT = """
The most recent {label} for this course already used the {noun} below. Do not repeat these, \
and do not ask the same thing in different words - cover other points from the passages \
instead:
{items}
"""

# Cap on how many previous {noun} are listed, so the prompt stays small.
MAX_PREVIOUS_ITEMS = 30

SUBMIT_FLASHCARDS_TOOL_NAME = "submit_flashcards"
SUBMIT_FLASHCARDS_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_FLASHCARDS_TOOL_NAME,
        "description": "Submit the generated flashcards.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "cards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "front": {"type": "string"},
                            "back": {"type": "string"},
                        },
                        "required": ["front", "back"],
                    },
                },
            },
            "required": ["title", "cards"],
        },
    },
}

GENERATE_FLASHCARDS_SYSTEM_PROMPT = """\
Generate flashcards for the course "{name}", with up to {count} cards - fewer only if the \
material below doesn't support that many good cards; never pad with filler. Ground every card in \
the passages below; do not invent facts. Keep each front a short prompt and each back a concise \
answer.

Passages:
{context}
{previous}
Call {tool} with the finished set.\
"""


async def study_tools_agent(state: AgentState, *, db: AsyncSession, course: Course) -> dict:
    if state["context_status"] == "empty_course":
        return results.success(state, AGENT_NAME, qa_agent.EMPTY_COURSE_MESSAGE)
    if state["context_status"] == "not_ready":
        return results.success(state, AGENT_NAME, qa_agent.NOT_READY_MESSAGE)

    resolution = await _resolve(state, course)
    if state["delegation_kind"] in ("quiz", "flashcards"):
        # The orchestrator named the item; the resolve call only fills in its details.
        resolution["study_tool_type"] = state["delegation_kind"]

    if resolution["study_tool_type"] == "unclear":
        return results.clarify(
            AGENT_NAME, "quiz vs flashcards is ambiguous", "Would you like a quiz or flashcards?"
        )

    if not resolution["covered"]:
        kind = resolution["study_tool_type"]
        return results.success(state, AGENT_NAME, _refusal_message(), kind=kind)

    if resolution["study_tool_type"] == "quiz":
        _fill_stated_quiz_details(resolution, state)
        if not resolution.get("format") or not resolution.get("difficulty"):
            return results.clarify(
                AGENT_NAME,
                "quiz format/difficulty not specified",
                "What format would you like (multiple choice, true/false, short answer, or "
                "mixed) and what difficulty (easy, medium, or hard)?",
            )
        return await _generate_quiz(db, course, state, resolution)

    return await _generate_flashcards(db, course, state, resolution)


async def _resolve(state: AgentState, course: Course) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _resolve_prompt(
                course, state["retrieved_context"], state["delegation_kind"]
            ),
        },
        *({"role": role, "content": content} for role, content in state["messages"]),
        {"role": "user", "content": results.task_message(state)},
    ]
    reply = await llm.complete(messages, tools=[RESOLVE_TOOL], tool_choice="required")
    call = next((c for c in reply.tool_calls if c.name == RESOLVE_TOOL_NAME), None)
    return _parse_resolution(call.arguments if call else {})


def _fill_stated_quiz_details(resolution: dict[str, Any], state: AgentState) -> None:
    """Fills in a quiz format/difficulty the student clearly stated but the model missed."""
    text = " ".join(
        [state["delegation_focus"], state["raw_message"], state["request_message"]]
    ).lower()
    if not resolution.get("format"):
        formats = {name for name, pattern in STATED_FORMATS if re.search(pattern, text)}
        if len(formats) == 1:
            resolution["format"] = formats.pop()
    if not resolution.get("difficulty"):
        difficulties = set(re.findall(r"\b(easy|medium|hard)\b", text))
        if len(difficulties) == 1:
            resolution["difficulty"] = difficulties.pop()


def _parse_resolution(arguments: dict[str, Any]) -> dict[str, Any]:
    study_tool_type = arguments.get("study_tool_type")
    if study_tool_type not in ("quiz", "flashcards", "unclear"):
        study_tool_type = "unclear"
    return {
        "study_tool_type": study_tool_type,
        "covered": bool(arguments.get("covered", False)),
        "format": arguments.get("format") if arguments.get("format") in QUIZ_FORMATS else None,
        "difficulty": (
            arguments.get("difficulty")
            if arguments.get("difficulty") in QUIZ_DIFFICULTIES
            else None
        ),
        "count": arguments.get("count") if isinstance(arguments.get("count"), int) else None,
    }


async def _generate_quiz(
    db: AsyncSession, course: Course, state: AgentState, resolution: dict[str, Any]
) -> dict:
    count = resolution["count"] or DEFAULT_QUIZ_COUNT
    quiz_format, difficulty = resolution["format"], resolution["difficulty"]
    messages = [
        {
            "role": "system",
            "content": GENERATE_QUIZ_SYSTEM_PROMPT.format(
                name=course.name,
                format=quiz_format,
                difficulty=difficulty,
                count=count,
                context=state["retrieved_context"],
                previous=await _previous_quiz_questions(db, course.id),
                tool=SUBMIT_QUIZ_TOOL_NAME,
            ),
        },
        {"role": "user", "content": results.task_message(state)},
    ]
    reply = await llm.complete(messages, tools=[SUBMIT_QUIZ_TOOL], tool_choice="required")
    call = next((c for c in reply.tool_calls if c.name == SUBMIT_QUIZ_TOOL_NAME), None)
    questions = ((call.arguments.get("questions") if call else None) or [])[:count]
    if not questions:
        logger.warning("Course %s: study_tools_agent produced no quiz questions", course.id)
        return results.success(state, AGENT_NAME, GENERATION_FAILED_MESSAGE, kind="quiz")

    title = str((call.arguments.get("title") if call else None) or f"{course.name} Quiz")
    quiz = Quiz(
        course_id=course.id,
        title=title,
        format=QuizFormat(quiz_format),
        difficulty=QuizDifficulty(difficulty),
    )
    db.add(quiz)
    await db.flush()  # need quiz.id for the question rows

    for order, question in enumerate(questions):
        question_type = _question_type(quiz_format, question.get("question_type"))
        options = question.get("options") if question_type == QuestionType.mcq else None
        db.add(
            QuizQuestion(
                quiz_id=quiz.id,
                question_text=str(question.get("question_text") or "").strip(),
                question_type=question_type,
                options=options if isinstance(options, list) else None,
                correct_answer=str(question.get("correct_answer") or "").strip(),
                order=order,
            )
        )
    # flush, not commit: the chat endpoint commits the whole turn at once.
    await db.flush()

    generated = len(questions)
    summary = (
        f"Quiz created: {generated} {quiz_format} question(s), {difficulty} difficulty, "
        "saved to Study Tools."
    )
    if generated < count:
        summary += _shortfall_note(generated, count)
    content = f'Generated quiz "{title}": ' + "; ".join(
        str(q.get("question_text", "")) for q in questions
    )
    return results.success(state, AGENT_NAME, summary, content=content, kind="quiz")


async def _generate_flashcards(
    db: AsyncSession, course: Course, state: AgentState, resolution: dict[str, Any]
) -> dict:
    count = resolution["count"] or DEFAULT_FLASHCARD_COUNT
    messages = [
        {
            "role": "system",
            "content": GENERATE_FLASHCARDS_SYSTEM_PROMPT.format(
                name=course.name,
                count=count,
                context=state["retrieved_context"],
                previous=await _previous_flashcard_fronts(db, course.id),
                tool=SUBMIT_FLASHCARDS_TOOL_NAME,
            ),
        },
        {"role": "user", "content": results.task_message(state)},
    ]
    reply = await llm.complete(messages, tools=[SUBMIT_FLASHCARDS_TOOL], tool_choice="required")
    call = next((c for c in reply.tool_calls if c.name == SUBMIT_FLASHCARDS_TOOL_NAME), None)
    cards = ((call.arguments.get("cards") if call else None) or [])[:count]
    if not cards:
        logger.warning("Course %s: study_tools_agent produced no flashcards", course.id)
        return results.success(state, AGENT_NAME, GENERATION_FAILED_MESSAGE, kind="flashcards")

    title = str((call.arguments.get("title") if call else None) or f"{course.name} Flashcards")
    flashcard_set = FlashcardSet(course_id=course.id, title=title)
    db.add(flashcard_set)
    await db.flush()  # need flashcard_set.id for the card rows

    for order, card in enumerate(cards):
        db.add(
            Flashcard(
                flashcard_set_id=flashcard_set.id,
                front_text=str(card.get("front") or "").strip(),
                back_text=str(card.get("back") or "").strip(),
                order=order,
            )
        )
    # flush, not commit: the chat endpoint commits the whole turn at once.
    await db.flush()

    generated = len(cards)
    summary = f"Flashcards created: {generated} card(s), saved to Study Tools."
    if generated < count:
        summary += _shortfall_note(generated, count)
    content = f'Generated flashcard set "{title}" with {generated} cards.'
    return results.success(state, AGENT_NAME, summary, content=content, kind="flashcards")


def _question_type(quiz_format: str, provided: Any) -> QuestionType:
    if quiz_format != QuizFormat.mixed.value:
        return QuestionType(quiz_format)
    if provided in QUESTION_TYPES:
        return QuestionType(provided)
    return QuestionType.short_answer


async def _previous_quiz_questions(db: AsyncSession, course_id: int) -> str:
    """The most recent quiz's question texts, as a "don't repeat these" block for the prompt."""
    quiz_id = await db.scalar(
        # By id: rows created in one transaction share the same created_at.
        select(Quiz.id).where(Quiz.course_id == course_id).order_by(Quiz.id.desc()).limit(1)
    )
    if quiz_id is None:
        return ""
    questions = await db.scalars(
        select(QuizQuestion.question_text)
        .where(QuizQuestion.quiz_id == quiz_id)
        .order_by(QuizQuestion.order)
        .limit(MAX_PREVIOUS_ITEMS)
    )
    return _previous_items_prompt("quiz", "questions", list(questions))


async def _previous_flashcard_fronts(db: AsyncSession, course_id: int) -> str:
    """The most recent set's card fronts, as a "don't repeat these" block for the prompt."""
    set_id = await db.scalar(
        select(FlashcardSet.id)
        .where(FlashcardSet.course_id == course_id)
        .order_by(FlashcardSet.id.desc())
        .limit(1)
    )
    if set_id is None:
        return ""
    fronts = await db.scalars(
        select(Flashcard.front_text)
        .where(Flashcard.flashcard_set_id == set_id)
        .order_by(Flashcard.order)
        .limit(MAX_PREVIOUS_ITEMS)
    )
    return _previous_items_prompt("flashcard set", "cards", list(fronts))


def _previous_items_prompt(label: str, noun: str, items: list[str]) -> str:
    listed = [item.strip() for item in items if item.strip()]
    if not listed:
        return ""
    return PREVIOUS_ITEMS_PROMPT.format(
        label=label, noun=noun, items="\n".join(f"- {item}" for item in listed)
    )


def _shortfall_note(generated: int, requested: int) -> str:
    return (
        f" I could only generate {generated} of the {requested} requested — upload more "
        "material on this topic for a fuller set."
    )


def _refusal_message() -> str:
    return (
        "I can't generate study material on that — it isn't covered by any of your uploaded "
        "materials. Upload material on this topic, or ask about something that's already covered."
    )


def _resolve_prompt(course: Course, context: str, assigned: str) -> str:
    return RESOLVE_SYSTEM_PROMPT.format(
        name=course.name,
        semester=course.semester,
        year=course.year,
        description=course.description or "(none)",
        context=context or "(no matching passages found)",
        tool=RESOLVE_TOOL_NAME,
        assigned=_assigned_item(assigned),
    )


def _assigned_item(assigned: str) -> str:
    if assigned == "quiz":
        return ASSIGNED_ITEM.format(
            article="a", item="quiz", quiz_details=", format and difficulty"
        )
    if assigned == "flashcards":
        return ASSIGNED_ITEM.format(article="a set of", item="flashcards", quiz_details="")
    return ""
