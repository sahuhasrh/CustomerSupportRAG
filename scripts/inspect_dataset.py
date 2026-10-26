
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "bitext_customer_support.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print("Downloading the official Bitext CSV...")
        df = pd.read_csv(DATASET_URL)
        df.to_csv(CSV_PATH, index=False)
    else:
        print("Using the existing raw CSV.")

    df = pd.read_csv(CSV_PATH)

    print("\n--- Dataset overview ---")
    print(f"File: {CSV_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    print("\n--- Data types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing values ---")
    print(df.isna().sum().to_string())

    print("\n--- Duplicate rows ---")
    print(f"Exact duplicates: {df.duplicated().sum():,}")

    print("\n--- Sample records ---")
    available = [
        col for col in
        ["instruction", "response", "category", "intent", "flags"]
        if col in df.columns
    ]
    print(df[available].head(3).to_string(index=False))

    print("\n--- Text length statistics ---")
    for col in ["instruction", "response"]:
        if col in df.columns:
            lengths = df[col].fillna("").astype(str).str.len()
            print(
                f"{col}: min={lengths.min()}, "
                f"median={lengths.median():.0f}, "
                f"max={lengths.max()}"
            )


if __name__ == "__main__":
    main()
