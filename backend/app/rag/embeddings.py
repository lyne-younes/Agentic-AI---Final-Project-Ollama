"""Embeddings for the RAG index, from a local Ollama model (fixed here, not configurable)."""

from app import ollama

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
# Must match the vector(1024) column. Changing the model re-indexes all materials on startup.
EMBEDDING_DIMENSIONS = 1024
_BATCH_SIZE = 32
_TIMEOUT_SECONDS = 300


class EmbeddingError(Exception):
    pass


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if ollama.is_downloading(EMBEDDING_MODEL):
        raise EmbeddingError(f"The embedding model {EMBEDDING_MODEL} is still downloading")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            response = await ollama.post(
                "/api/embed",
                {"model": EMBEDDING_MODEL, "input": batch},
                timeout=_TIMEOUT_SECONDS,
            )
        except ollama.OllamaError as exc:
            raise EmbeddingError(f"Ollama embeddings request failed: {exc}") from exc

        batch_vectors = response.get("embeddings") or []
        if len(batch_vectors) != len(batch) or any(
            len(vector) != EMBEDDING_DIMENSIONS for vector in batch_vectors
        ):
            raise EmbeddingError(
                f"{EMBEDDING_MODEL} didn't return {len(batch)} vectors of "
                f"{EMBEDDING_DIMENSIONS} numbers"
            )
        vectors.extend(batch_vectors)
    return vectors
