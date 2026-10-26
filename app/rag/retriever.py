
import os
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "customer_support_faq_1000",
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist."
        )

    return client


def retrieve_faqs(
    question: str,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[dict]:
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    model = get_embedding_model()
    client = get_qdrant_client()

    vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=score_threshold,
    )

    return [
        {
            "faq_id": point.payload.get("faq_id", ""),
            "question": point.payload.get("instruction", ""),
            "answer": point.payload.get("response", ""),
            "category": point.payload.get("category", ""),
            "intent": point.payload.get("intent", ""),
            "score": float(point.score),
        }
        for point in result.points
    ]
