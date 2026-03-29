from pathlib import Path
import argparse

from ultralytics import YOLO


DEFAULT_SEED = 42


def find_fold_yaml_files(kfold_dir: Path):
    """
    Locate fold YAML files.

    Expected structure:

        kfold/
            fold_1/
                fold_1.yaml
            fold_2/
                fold_2.yaml
    """

    return sorted(
        kfold_dir.glob("fold_*/*.yaml")
    )


def train_single_fold(
    yaml_file: Path,
    fold_index: int,
    weights: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device,
    project: str,
    run_name_prefix: str,
):
    """
    Train one fold.
    """

    print()
    print("=" * 50)
    print(f"Training Fold {fold_index}")
    print("=" * 50)

    model = YOLO(weights)

    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=f"{run_name_prefix}_{fold_index}",
        seed=DEFAULT_SEED,
        deterministic=True,
        workers=8,
        verbose=True,

        # augmentations
        hsv_h=0.02,
        hsv_s=0.9,
        hsv_v=0.8,
        scale=0.3,
        shear=10.0,
        erasing=0.3,
    )

    return results


def train_kfold(
    kfold_dir: str,
    weights: str = "yolo26m.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device="0",
    project: str = "runs/kfold",
    run_name_prefix: str = "fold",
    start_fold: int = 1,
):
    """
    Train all folds sequentially.

    Parameters
    ----------
    kfold_dir : str
        Directory containing fold datasets
    start_fold : int
        Resume training from specific fold
    """

    kfold_path = Path(kfold_dir)

    if not kfold_path.exists():
        raise FileNotFoundError(kfold_path)

    yaml_files = find_fold_yaml_files(kfold_path)

    if not yaml_files:
        raise RuntimeError("No fold YAML files found")

    print(f"Found {len(yaml_files)} folds")

    results = {}

    for fold_index, yaml_file in enumerate(yaml_files, start=1):

        if fold_index < start_fold:
            print(f"Skipping Fold {fold_index}")
            continue

        results[fold_index] = train_single_fold(
            yaml_file=yaml_file,
            fold_index=fold_index,
            weights=weights,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            run_name_prefix=run_name_prefix,
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO using K-Fold datasets"
    )

    parser.add_argument(
        "--kfold-dir",
        required=True,
        help="Path to k-fold dataset directory"
    )

    parser.add_argument(
        "--weights",
        default="yolo26m.pt"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--device",
        default="0"
    )

    parser.add_argument(
        "--project",
        default="runs/kfold"
    )

    parser.add_argument(
        "--run-name-prefix",
        default="fold"
    )

    parser.add_argument(
        "--start-fold",
        type=int,
        default=1
    )

    return parser.parse_args()


def main():

    args = parse_args()

    train_kfold(
        kfold_dir=args.kfold_dir,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        run_name_prefix=args.run_name_prefix,
        start_fold=args.start_fold,
    )


if __name__ == "__main__":
    main()