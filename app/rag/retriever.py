
import os
from functools import lru_cache

from qdrant_client import QdrantClient
from fastembed import TextEmbedding
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "customer_support_faq_1000",
)


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=api_key,
        timeout=30
    )

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

    vector = list(model.embed([question]))[0].tolist()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
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
