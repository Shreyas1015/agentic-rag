"""Dense + sparse embeddings for child chunks.

  dense  : openai/text-embedding-3-small via OpenRouter (1536-d), batched
  sparse : Qdrant/bm25 via FastEmbed (local, CPU)

Both flow into one Qdrant point per child chunk under named vectors
{"dense": ..., "bm25": ...} — see app.ingestion.upserter.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastembed import SparseTextEmbedding

from app.core.config import settings
from app.core.llm import get_async_openai_client
from app.ingestion.chunker import ChildNode

DENSE_BATCH_SIZE = 100  # OpenAI's recommended batch ceiling


@dataclass
class EmbeddedChunk:
    child: ChildNode
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        # First call downloads ~10 MB of model files into the FastEmbed cache.
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


async def embed_query_dense(text: str) -> list[float]:
    """Single-string dense embedding via OpenRouter — used at query time
    by both hybrid_search and the semantic cache."""
    client = get_async_openai_client()
    resp = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL, input=[text]
    )
    return resp.data[0].embedding


def embed_query_sparse(text: str) -> tuple[list[int], list[float]]:
    """Single-string BM25 sparse embedding via the local FastEmbed model."""
    bm25 = _get_bm25_model()
    sparse = next(iter(bm25.embed([text])))
    return sparse.indices.tolist(), sparse.values.tolist()


async def embed_chunks(
    children: list[ChildNode], *, dense_batch_size: int = DENSE_BATCH_SIZE
) -> list[EmbeddedChunk]:
    """Embed every child chunk twice (dense + sparse), preserving order."""
    if not children:
        return []

    client = get_async_openai_client()
    texts = [c.text for c in children]

    # Dense — batched calls to OpenRouter.
    dense_vectors: list[list[float]] = []
    for i in range(0, len(texts), dense_batch_size):
        batch = texts[i : i + dense_batch_size]
        resp = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL, input=batch
        )
        # OpenAI/OpenRouter return embeddings in input order.
        dense_vectors.extend(item.embedding for item in resp.data)

    # Sparse — local CPU model. embed() is a generator over numpy SparseEmbedding.
    bm25_model = _get_bm25_model()
    sparse_embeddings = list(bm25_model.embed(texts))

    return [
        EmbeddedChunk(
            child=child,
            dense=dense,
            sparse_indices=sparse.indices.tolist(),
            sparse_values=sparse.values.tolist(),
        )
        for child, dense, sparse in zip(
            children, dense_vectors, sparse_embeddings, strict=True
        )
    ]
