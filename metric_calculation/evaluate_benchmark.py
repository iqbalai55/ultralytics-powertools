import pandas as pd
from pathlib import Path
import argparse
import sys


# =========================
# DEFAULT CONFIG
# =========================
DEFAULT_WEIGHTS = [0.0, 0.0, 0.1, 0.9]  # [precision, recall, mAP50, mAP50-95]


# =========================
# UTILITIES
# =========================
def safe_get_column(df, candidates):
    """Return first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_f1(p, r):
    denom = p + r
    f1 = (2 * p * r) / denom
    return f1.fillna(0.0)


def compute_fitness(df, cols, weights):
    return (
        weights[0] * df[cols["precision"]]
        + weights[1] * df[cols["recall"]]
        + weights[2] * df[cols["map50"]]
        + weights[3] * df[cols["map5095"]]
    )


# =========================
# CORE
# =========================
def find_best_epoch(results_csv, weights, custom_cols=None):

    df = pd.read_csv(results_csv)

    # Auto-detect columns (Ultralytics detect/segment compatibility)
    cols = {
        "precision": safe_get_column(df, [
            "metrics/precision(B)", "metrics/precision(M)"
        ]),
        "recall": safe_get_column(df, [
            "metrics/recall(B)", "metrics/recall(M)"
        ]),
        "map50": safe_get_column(df, [
            "metrics/mAP50(B)", "metrics/mAP50(M)"
        ]),
        "map5095": safe_get_column(df, [
            "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"
        ]),
    }

    # Override if user provides custom mapping
    if custom_cols:
        cols.update(custom_cols)

    # Validate
    for k, v in cols.items():
        if v is None:
            raise ValueError(f"Missing column for {k}")

    # Compute metrics
    df["f1_score"] = compute_f1(
        df[cols["precision"]],
        df[cols["recall"]],
    )

    df["fitness"] = compute_fitness(df, cols, weights)

    # Best epoch
    best_idx = df["fitness"].idxmax()
    row = df.loc[best_idx]

    return {
        "epoch": int(row["epoch"]),
        "precision": row[cols["precision"]],
        "recall": row[cols["recall"]],
        "f1_score": row["f1_score"],
        "fitness": row["fitness"],
        "mAP50": row[cols["map50"]],
        "mAP50-95": row[cols["map5095"]],
    }


def summarize(root_dir, weights, recursive=False):

    root = Path(root_dir)
    rows = []

    pattern = "**/results.csv" if recursive else "*/results.csv"

    for results_csv in root.glob(pattern):

        model_name = results_csv.parent.name

        try:
            metrics = find_best_epoch(results_csv, weights)
            metrics["model"] = model_name
            rows.append(metrics)

            print(f"Processed: {model_name}")

        except Exception as e:
            print(f"Error in {model_name}: {e}")

    if not rows:
        print("No valid results found.")
        return None

    df = pd.DataFrame(rows)

    return df[
        [
            "model",
            "epoch",
            "precision",
            "recall",
            "f1_score",
            "fitness",
            "mAP50",
            "mAP50-95",
        ]
    ]


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ultralytics Benchmark Summarizer"
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing runs",
    )

    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=DEFAULT_WEIGHTS,
        metavar=("P", "R", "MAP50", "MAP5095"),
        help="Weights for fitness calculation",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for results.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_summary",
        help="Output file prefix",
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    summary = summarize(
        root_dir=args.root,
        weights=args.weights,
        recursive=args.recursive,
    )

    if summary is None:
        sys.exit(0)

    summary = summary.sort_values(
        by="fitness",
        ascending=False,
    ).round(4)

    print("\n=== BENCHMARK (BEST FITNESS) ===\n")
    print(f"Weights: {args.weights}\n")
    print(summary.to_string(index=False))

    # Save
    csv_path = f"{args.output}.csv"
    xlsx_path = f"{args.output}.xlsx"

    summary.to_csv(csv_path, index=False)
    summary.to_excel(xlsx_path, index=False)

    print("\nSaved:")
    print(f" - {csv_path}")
    print(f" - {xlsx_path}")


if __name__ == "__main__":
    main()