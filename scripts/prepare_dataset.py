
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bitext_customer_support.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "customer_support_clean.csv"
)

REQUIRED_COLUMNS = [
    "instruction",
    "response",
    "category",
    "intent",
    "flags",
]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep the useful question, answer and metadata fields.
    df = df[REQUIRED_COLUMNS].copy()

    # Normalize whitespace without altering placeholder values.
    for column in REQUIRED_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Remove records without a usable question or answer.
    before = len(df)
    df = df[
        df["instruction"].ne("")
        & df["response"].ne("")
    ].copy()
    removed_empty = before - len(df)

    # Remove duplicate question-answer pairs if present.
    before = len(df)
    df = df.drop_duplicates(
        subset=["instruction", "response"]
    ).copy()
    removed_duplicates = before - len(df)

    # Stable identifiers help us trace retrieved sources.
    df.insert(
        0,
        "faq_id",
        [f"faq_{i:05d}" for i in range(1, len(df) + 1)],
    )

    # Text to embed: question plus its corresponding answer.
    df["search_text"] = (
        "Customer question: "
        + df["instruction"]
        + "\nSuggested response: "
        + df["response"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset preparation complete.")
    print(f"Input records: {before + removed_empty:,}")
    print(f"Removed empty records: {removed_empty:,}")
    print(f"Removed duplicate pairs: {removed_duplicates:,}")
    print(f"Output records: {len(df):,}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Output columns: {df.columns.tolist()}")
    print("\nSample search text:")
    print(df.iloc[0]["search_text"])


if __name__ == "__main__":
    main()
