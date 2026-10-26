
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models


ROOT_DIR = Path(__file__).resolve().parents[1]

EMBEDDING_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "subset_embeddings"
    / "faq_embeddings.npy"
)
METADATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "subset_embeddings"
    / "embedding_metadata.csv"
)

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "customer_support_faq_1000"
VECTOR_SIZE = 384
BATCH_SIZE = 100


def main():
    if not EMBEDDING_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Subset embeddings or metadata are missing. "
            "Generate the subset embeddings before indexing."
        )

    embeddings = np.load(EMBEDDING_FILE)
    metadata = pd.read_csv(METADATA_FILE).fillna("")

    if embeddings.shape != (1000, VECTOR_SIZE):
        raise ValueError(
            f"Expected embeddings of shape (1000, 384), "
            f"but found {embeddings.shape}."
        )

    if len(metadata) != len(embeddings):
        raise ValueError("Embedding and metadata counts do not match.")

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    try:
        # This script intentionally recreates only the subset collection.
        if client.collection_exists(COLLECTION_NAME):
            print(f"Deleting existing subset collection: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        for start in range(0, len(metadata), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(metadata))
            points = []

            for index in range(start, end):
                row = metadata.iloc[index]

                payload = {
                    key: str(row[key])
                    for key in metadata.columns
                }

                points.append(
                    models.PointStruct(
                        id=index + 1,
                        vector=embeddings[index].tolist(),
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )
            print(f"Indexed {end}/{len(metadata)} records")

        info = client.get_collection(COLLECTION_NAME)
        print("\nCollection setup complete.")
        print("Collection:", COLLECTION_NAME)
        print("Points:", info.points_count)
        print("Status:", info.status)

        if info.points_count != len(metadata):
            raise RuntimeError("The final Qdrant point count is incorrect.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
