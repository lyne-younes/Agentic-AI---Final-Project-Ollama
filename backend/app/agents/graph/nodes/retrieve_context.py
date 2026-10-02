"""Retrieval step for the Q&A, study tools and summary agents.

Searches the course materials, or with `use_previous_result` reuses the previous agent's output.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph.state import AgentState
from app.models import Course, IndexingStatus, Material
from app.rag import search

SEARCH_RESULTS = 8


async def retrieve_context(state: AgentState, *, db: AsyncSession, course: Course) -> dict:
    if state["use_previous_result"] and state["agent_results"]:
        source = state["agent_results"][-1]["content"]
        return {"context_status": "ok", "retrieved_context": source}

    materials = list(
        await db.scalars(select(Material).where(Material.course_id == course.id).limit(1))
    )
    if not materials:
        return {"context_status": "empty_course", "retrieved_context": ""}
    any_indexed = await db.scalar(
        select(Material.id)
        .where(Material.course_id == course.id, Material.indexing_status == IndexingStatus.indexed)
        .limit(1)
    )
    if any_indexed is None:
        return {"context_status": "not_ready", "retrieved_context": ""}

    query = state["retrieval_query"] or state["raw_message"]
    matches = await search.search_materials(db, course.id, query, limit=SEARCH_RESULTS)
    passages = "\n\n".join(
        f'From "{match.material_title}":\n{match.chunk_text}' for match in matches
    )
    return {"context_status": "ok", "retrieved_context": passages}
