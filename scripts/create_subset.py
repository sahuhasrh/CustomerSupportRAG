
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "processed" / "customer_support_clean.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "customer_support_subset_1000.csv"

TARGET_SIZE = 1000
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_FILE)

    if "intent" not in df.columns:
        raise ValueError("The dataset must contain an 'intent' column.")

    intent_counts = df["intent"].value_counts(dropna=False)

    if len(intent_counts) > TARGET_SIZE:
        raise ValueError(
            f"Cannot guarantee every intent in {TARGET_SIZE} rows: "
            f"found {len(intent_counts)} intents."
        )

    # Allocate at least one row to every intent, then distribute
    # the remaining rows proportionally to each intent's frequency.
    proportions = intent_counts / intent_counts.sum()
    ideal_allocations = proportions * TARGET_SIZE

    allocations = ideal_allocations.apply(int)
    allocations = allocations.clip(lower=1)

    # Adjust allocations to total exactly TARGET_SIZE.
    while allocations.sum() < TARGET_SIZE:
        eligible = intent_counts - allocations
        eligible = eligible[eligible > 0]
        if eligible.empty:
            raise ValueError("Not enough source records to reach target size.")

        priorities = ideal_allocations.loc[eligible.index] - allocations.loc[eligible.index]
        allocations.loc[priorities.idxmax()] += 1

    while allocations.sum() > TARGET_SIZE:
        eligible = allocations[allocations > 1]
        if eligible.empty:
            raise ValueError("Cannot reduce allocations to the target size.")

        priorities = allocations.loc[eligible.index] - ideal_allocations.loc[eligible.index]
        allocations.loc[priorities.idxmax()] -= 1

    sampled_parts = []

    for intent, count in allocations.items():
        group = df[df["intent"].isna()] if pd.isna(intent) else df[df["intent"] == intent]
        sampled_parts.append(
            group.sample(n=int(count), random_state=RANDOM_SEED)
        )

    subset = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    subset.to_csv(OUTPUT_FILE, index=False)

    print("Subset created successfully.")
    print(f"Original records: {len(df):,}")
    print(f"Subset records:   {len(subset):,}")
    print(f"Distinct intents: {subset['intent'].nunique()}")
    print(f"Distinct categories: {subset['category'].nunique()}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nRecords per intent in subset:")
    print(subset["intent"].value_counts().sort_index().to_string())

    missing_intents = set(df["intent"].dropna().unique()) - set(
        subset["intent"].dropna().unique()
    )

    if missing_intents:
        raise RuntimeError(f"Some intents were lost: {sorted(missing_intents)}")

    print("\nValidation passed: all intents are represented.")


if __name__ == "__main__":
    main()
