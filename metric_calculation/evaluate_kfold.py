from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def find_fold_directories(models_dir: Path) -> List[Path]:
    """
    Locate all fold directories.

    Expected structure:
        models/
            fold_1/
            fold_2/
            fold_3/

    Returns
    -------
    List[Path]
    """

    return sorted(
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("fold")
    )


def load_model(model_path: Path) -> YOLO:
    """
    Load YOLO model.

    Raises
    ------
    FileNotFoundError
    """

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    return YOLO(model_path)


def run_validation(
    model: YOLO,
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str
) -> Dict:
    """
    Run YOLO validation.

    Returns
    -------
    Dict
        Metrics dictionary
    """

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )

    return {
        "accuracy": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(metrics.box.f1),
    }


def save_metrics_csv(metrics: Dict, output_file: Path) -> None:
    """
    Save metrics to CSV.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


# -----------------------------------------------------------------------------
# Core Evaluation
# -----------------------------------------------------------------------------

def evaluate_kfold(
    models_dir: Path,
    output_dir: Path,
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str
) -> None:
    """
    Evaluate all folds.
    """

    fold_dirs = find_fold_directories(models_dir)

    if not fold_dirs:
        raise RuntimeError("No fold directories found")

    all_results: List[Dict] = []

    for fold_dir in fold_dirs:

        fold_name = fold_dir.name
        model_path = fold_dir / "weights" / "best.pt"

        if not model_path.exists():
            continue

        model = load_model(model_path)

        metrics = run_validation(
            model=model,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=device
        )

        metrics["fold"] = fold_name

        save_metrics_csv(
            metrics,
            output_dir / fold_name / "metrics.csv"
        )

        all_results.append(metrics)

    if not all_results:
        raise RuntimeError("No results generated")

    compute_macro_average(all_results, output_dir)


# -----------------------------------------------------------------------------
# Macro Average
# -----------------------------------------------------------------------------

def compute_macro_average(
    results: List[Dict],
    output_dir: Path
) -> None:
    """
    Compute macro-average metrics.
    """

    macro_metrics = {
        "fold": "MACRO_AVG",
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "f1": np.mean([r["f1"] for r in results]),
    }

    summary_file = output_dir / "kfold_summary.csv"

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(macro_metrics)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="YOLO K-Fold Evaluation"
    )

    parser.add_argument(
        "--models",
        required=True,
        help="Path to k-fold models directory"
    )

    parser.add_argument(
        "--output",
        default="result/kfold",
        help="Output directory"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML file"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--device",
        default="0"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    evaluate_kfold(
        models_dir=Path(args.models),
        output_dir=Path(args.output),
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device
    )


if __name__ == "__main__":
    main()