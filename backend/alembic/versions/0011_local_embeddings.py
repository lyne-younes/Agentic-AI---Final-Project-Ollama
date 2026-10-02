"""Local embeddings: material_chunks.embedding becomes vector(1024) for Ollama's
qwen3-embedding:0.6b (was vector(1536) for OpenAI's text-embedding-3-large)

Vectors from the old model can't be converted, so the chunks are deleted and their materials set
back to pending. On startup the backend re-indexes every pending material with the new model
(indexing.reindex_outdated_materials); the original text in materials.raw_text is untouched.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-25 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _resize(dimensions: int) -> None:
    op.execute("DELETE FROM material_chunks")
    op.execute("UPDATE materials SET indexing_status = 'pending' WHERE indexing_status = 'indexed'")
    op.drop_index("ix_material_chunks_embedding", table_name="material_chunks")
    op.execute(f"ALTER TABLE material_chunks ALTER COLUMN embedding TYPE vector({dimensions})")
    op.create_index(
        "ix_material_chunks_embedding",
        "material_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def upgrade() -> None:
    _resize(1024)


def downgrade() -> None:
    _resize(1536)
