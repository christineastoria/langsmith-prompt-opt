"""
Creates combined core + edge datasets for the with-edge optimizer round.

Combines the core train/val splits (from train.jsonl / val.jsonl) with a
stratified split of the edge examples, then uploads them to LangSmith as
separate datasets so they don't overwrite the core splits.

LangSmith dataset names:
    shopping-concierge-routing-with-edge-train
    shopping-concierge-routing-with-edge-val

Usage:
    uv run python src/eval/dataset_with_edge.py
    uv run python src/eval/dataset_with_edge.py --replace
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from dataset_edge import EDGE_EXAMPLES

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"
BASE_NAME = "shopping-concierge-routing-with-edge"

TRAIN_NAME = f"{BASE_NAME}-train"
VAL_NAME   = f"{BASE_NAME}-val"


# ---------------------------------------------------------------------------
# Split edge examples stratified by difficulty category
# ---------------------------------------------------------------------------

def split_edge(examples: list[dict], train: float = 0.6, val: float = 0.2, seed: int = 42) -> tuple:
    """
    Stratified split of edge examples by difficulty tag.
    Categories with only 1 example go to train to maximise optimizer signal.
    Categories with 2+ examples are split proportionally.
    """
    random.seed(seed)

    by_difficulty = defaultdict(list)
    for ex in examples:
        difficulty = ex["outputs"].get("difficulty", "unknown")
        by_difficulty[difficulty].append(ex)

    train_ex, val_ex, test_ex = [], [], []

    for difficulty, group in by_difficulty.items():
        random.shuffle(group)
        n = len(group)
        if n == 1:
            # Single example for this difficulty — put in train for optimizer signal
            train_ex.extend(group)
        else:
            n_train = max(1, round(n * train))
            n_val   = max(0, round(n * val))
            train_ex.extend(group[:n_train])
            val_ex.extend(group[n_train:n_train + n_val])
            test_ex.extend(group[n_train + n_val:])

    return train_ex, val_ex, test_ex


# ---------------------------------------------------------------------------
# Load core splits from JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            examples.append({
                "inputs":  ex["inputs"],
                "outputs": ex["outputs"],
            })
    return examples


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_split(client: Client, name: str, examples: list[dict], replace: bool) -> None:
    if client.has_dataset(dataset_name=name):
        if not replace:
            print(f"  '{name}' already exists — skipping. Pass --replace to overwrite.")
            return
        client.delete_dataset(dataset_name=name)
        print(f"  Deleted existing '{name}'")
    client.create_dataset(
        dataset_name=name,
        description=f"Shopping concierge routing — core + edge examples ({name.split('-')[-1]} split)",
    )
    client.create_examples(
        inputs=[ex["inputs"]  for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_name=name,
    )
    print(f"  Uploaded {len(examples)} examples → '{name}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Delete and re-upload existing datasets")
    args = parser.parse_args()

    # Load core splits
    core_train = load_jsonl(DATA_DIR / "train.jsonl")
    core_val   = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"Core: {len(core_train)} train / {len(core_val)} val")

    # Split edge examples
    edge_train, edge_val, edge_test = split_edge(EDGE_EXAMPLES)
    print(f"Edge: {len(edge_train)} train / {len(edge_val)} val / {len(edge_test)} test")

    # Print edge train difficulty coverage
    print("\nEdge train difficulties:")
    for ex in sorted(edge_train, key=lambda e: e["outputs"].get("difficulty", "")):
        print(f"  {ex['outputs'].get('difficulty', '?'):35s}  {ex['inputs']['query'][:60]}")

    print("\nEdge val difficulties:")
    for ex in sorted(edge_val, key=lambda e: e["outputs"].get("difficulty", "")):
        print(f"  {ex['outputs'].get('difficulty', '?'):35s}  {ex['inputs']['query'][:60]}")

    # Combine
    combined_train = core_train + edge_train
    combined_val   = core_val   + edge_val
    print(f"\nCombined: {len(combined_train)} train / {len(combined_val)} val")

    # Upload
    print("\nUploading to LangSmith...")
    client = Client()
    upload_split(client, TRAIN_NAME, combined_train, replace=args.replace)
    upload_split(client, VAL_NAME,   combined_val,   replace=args.replace)
    print("\nDone.")
