
from pathlib import Path
import time

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = (
    ROOT / "data" / "processed" / "embeddings" / "faq_embeddings.npy"
)
METADATA_PATH = (
    ROOT / "data" / "processed" / "embeddings" / "embedding_metadata.csv"
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "customer_support_faq"
VECTOR_SIZE = 384
UPLOAD_BATCH_SIZE = 128


def main() -> None:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Embedding or metadata file missing. Complete Step 8 first."
        )

    vectors = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    metadata = pd.read_csv(METADATA_PATH).fillna("")

    if vectors.shape != (len(metadata), VECTOR_SIZE):
        raise ValueError(
            f"Shape mismatch: vectors={vectors.shape}, "
            f"metadata rows={len(metadata)}"
        )

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # Recreate the collection to prevent stale or duplicate points.
    # This deletes the previous collection if it exists.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

    total = len(metadata)
    started = time.perf_counter()

    for start in range(0, total, UPLOAD_BATCH_SIZE):
        end = min(start + UPLOAD_BATCH_SIZE, total)
        batch_points = []

        for i in range(start, end):
            row = metadata.iloc[i]

            payload = {
                "faq_id": str(row["faq_id"]),
                "instruction": str(row["instruction"]),
                "response": str(row["response"]),
                "category": str(row["category"]),
                "intent": str(row["intent"]),
                "flags": str(row["flags"]),
            }

            batch_points.append(
                models.PointStruct(
                    id=i + 1,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points,
            wait=True,
        )

        print(f"Uploaded {end:,}/{total:,} records")

    info = client.get_collection(COLLECTION_NAME)

    print("\nIndexing complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Expected records: {total:,}")
    print(f"Stored points: {info.points_count:,}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.1f}")

    client.close()


if __name__ == "__main__":
    main()
