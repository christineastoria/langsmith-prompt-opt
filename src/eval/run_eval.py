"""
Run the routing eval against a LangSmith dataset split.

    # Evaluate baseline prompt against val set
    uv run python src/eval/run_eval.py --prompt baseline --split val

    # Evaluate optimised prompt against test set
    uv run python src/eval/run_eval.py --prompt optimized --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langsmith import evaluate

from eval.evaluator import task_completeness, critical_agents_called, sequence_respected
from eval.run_function import make_run_fn

load_dotenv()

DATASET_NAME = "shopping-concierge-routing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="baseline", help="Prompt variant (baseline | optimized)")
    parser.add_argument("--split", default="val", help="Dataset split (train | val | test)")
    parser.add_argument("--prefix", default=None, help="Experiment name prefix")
    parser.add_argument("--suffix", default="", help="Dataset name suffix, e.g. '-v2' to use shopping-concierge-routing-v2-train")
    args = parser.parse_args()

    dataset_name = f"{DATASET_NAME}{args.suffix}-{args.split}"
    prefix = args.prefix or f"{args.prompt}-{args.split}{args.suffix}"

    print(f"\nEvaluating '{args.prompt}' prompt on '{dataset_name}'...")

    results = evaluate(
        make_run_fn(args.prompt),
        data=dataset_name,
        evaluators=[task_completeness, critical_agents_called, sequence_respected],
        experiment_prefix=prefix,
        max_concurrency=1,
    )

    # Print summary
    print(f"\n--- Results: {prefix} ---")
    df = results.to_pandas()
    score_cols = [c for c in df.columns if any(m in c for m in ["task_completeness", "critical_agents_called", "sequence_respected"])]
    print(df[score_cols].describe().to_string())


if __name__ == "__main__":
    main()
