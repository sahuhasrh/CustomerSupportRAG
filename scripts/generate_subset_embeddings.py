
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT_DIR / "data" / "processed" / "customer_support_subset_1000.csv"
)
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "subset_embeddings"
EMBEDDING_FILE = OUTPUT_DIR / "faq_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "embedding_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Subset not found: {INPUT_FILE}\n"
            "Run scripts\\create_subset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 records, but found {len(df)}. "
            "Check the subset CSV before continuing."
        )

    required_columns = {"faq_id", "instruction", "response"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    texts = (
        "Question: "
        + df["instruction"].fillna("").astype(str)
        + "\nAnswer: "
        + df["response"].fillna("").astype(str)
    ).tolist()

    print(f"Records: {len(texts)}")
    print(f"Loading model: {MODEL_NAME}")

    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Model loaded in {time.perf_counter() - start:.2f} seconds.")

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(df), 384):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    np.save(EMBEDDING_FILE, embeddings)

    metadata_columns = [
        column
        for column in [
            "faq_id",
            "instruction",
            "response",
            "category",
            "intent",
            "flags",
        ]
        if column in df.columns
    ]

    df[metadata_columns].to_csv(METADATA_FILE, index=False)

    print(f"\nEmbedding generation: {time.perf_counter() - start:.2f} seconds.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embeddings saved: {EMBEDDING_FILE}")
    print(f"Metadata saved: {METADATA_FILE}")
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
