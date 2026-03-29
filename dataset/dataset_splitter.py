
from __future__ import annotations

import shutil
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from collections import Counter
from tqdm import tqdm


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class YOLODatasetSplitter:
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        yaml_file: str | Path,
        output_dir: str | Path = "splits",
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.yaml_file = Path(yaml_file)
        self.output_dir = Path(output_dir)
        self.seed = seed

        random.seed(seed)

        self.classes = self._load_classes()
        self.labels = self._collect_labels()
        self.images = self._collect_images()

        self.df = self._build_label_dataframe()

    def _load_classes(self) -> Dict:
        with open(self.yaml_file, encoding="utf8") as f:
            data = yaml.safe_load(f)

        return data["names"]

    def _collect_labels(self) -> List[Path]:
        return sorted(self.labels_dir.rglob("*.txt"))

    def _collect_images(self) -> List[Path]:
        images: List[Path] = []

        for ext in SUPPORTED_EXTENSIONS:
            images.extend(sorted(self.images_dir.rglob(f"*{ext}")))

        return images

    def _build_label_dataframe(self) -> pd.DataFrame:
        cls_idx = sorted(self.classes.keys())

        index = [label.stem for label in self.labels]

        df = pd.DataFrame(columns=cls_idx, index=index)

        for label in self.labels:

            counter = Counter()

            with open(label) as f:
                lines = f.readlines()

            for line in lines:
                cls = int(line.split(" ", 1)[0])
                counter[cls] += 1

            df.loc[label.stem] = counter

        return df.fillna(0.0)

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ):

        assert train_ratio + val_ratio + test_ratio == 1.0

        train_idx, temp_idx = train_test_split(
            self.df.index,
            test_size=(1 - train_ratio),
            random_state=self.seed,
            shuffle=True,
        )

        val_size = val_ratio / (val_ratio + test_ratio)

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_size),
            random_state=self.seed,
            shuffle=True,
        )

        split_map = {}

        for idx in train_idx:
            split_map[idx] = "train"

        for idx in val_idx:
            split_map[idx] = "val"

        for idx in test_idx:
            split_map[idx] = "test"

        self._copy_files(split_map, mode="standard")

        print("Dataset split completed.")
    
    def stratified_kfold_split(self, k: int = 5):
        """
        Stratified K-Fold split based on dominant class per image.
        """

        # Determine dominant class per sample
        dominant_class = self.df.idxmax(axis=1)

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=self.seed,
        )

        folds_df = pd.DataFrame(
            index=self.df.index,
            columns=[f"fold_{i}" for i in range(1, k + 1)],
        )

        for i, (train_idx, val_idx) in enumerate(
            skf.split(self.df, dominant_class),
            start=1,
        ):

            folds_df.iloc[train_idx, i - 1] = "train"
            folds_df.iloc[val_idx, i - 1] = "val"

        self._create_kfold_directories(folds_df)

        print("Stratified K-Fold split completed.")

        return folds_df

    def kfold_split(self, k: int = 5):

        kf = KFold(
            n_splits=k,
            shuffle=True,
            random_state=self.seed,
        )

        folds_df = pd.DataFrame(
            index=self.df.index,
            columns=[f"fold_{i}" for i in range(1, k + 1)],
        )

        for i, (train_idx, val_idx) in enumerate(kf.split(self.df), start=1):

            folds_df.iloc[train_idx, i - 1] = "train"
            folds_df.iloc[val_idx, i - 1] = "val"

        self._create_kfold_directories(folds_df)

        print("K-Fold split completed.")

        return folds_df


    def _copy_files(
        self,
        split_map: Dict[str, str],
        mode: str,
    ):

        date = datetime.date.today().isoformat()

        save_path = self.output_dir / f"{date}_{mode}"
        save_path.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:

            (save_path / split / "images").mkdir(
                parents=True,
                exist_ok=True,
            )

            (save_path / split / "labels").mkdir(
                parents=True,
                exist_ok=True,
            )

        for image in tqdm(self.images):

            stem = image.stem

            if stem not in split_map:
                continue

            split = split_map[stem]

            label = self.labels_dir / f"{stem}.txt"

            shutil.copy(
                image,
                save_path / split / "images" / image.name,
            )

            shutil.copy(
                label,
                save_path / split / "labels" / label.name,
            )

        self._write_yaml(save_path)


    def _create_kfold_directories(
        self,
        folds_df: pd.DataFrame,
    ):

        date = datetime.date.today().isoformat()

        save_path = self.output_dir / f"{date}_kfold"
        save_path.mkdir(parents=True, exist_ok=True)

        for fold in folds_df.columns:

            split_dir = save_path / fold

            for subset in ["train", "val"]:

                (split_dir / subset / "images").mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (split_dir / subset / "labels").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            dataset_yaml = split_dir / f"{fold}.yaml"

            with open(dataset_yaml, "w") as f:

                yaml.safe_dump(
                    {
                        "path": split_dir.as_posix(),
                        "train": "train",
                        "val": "val",
                        "names": self.classes,
                    },
                    f,
                )

        for image in tqdm(self.images):

            stem = image.stem

            if stem not in folds_df.index:
                continue

            for fold in folds_df.columns:

                subset = folds_df.loc[stem, fold]

                if pd.isna(subset):
                    continue

                label = self.labels_dir / f"{stem}.txt"

                shutil.copy(
                    image,
                    save_path / fold / subset / "images" / image.name,
                )

                if label.exists():

                    shutil.copy(
                        label,
                        save_path / fold / subset / "labels" / label.name,
                    )


    def _write_yaml(self, path: Path):

        dataset_yaml = path / "dataset.yaml"

        with open(dataset_yaml, "w") as f:

            yaml.safe_dump(
                {
                    "path": path.as_posix(),
                    "train": "train",
                    "val": "val",
                    "test": "test",
                    "names": self.classes,
                },
                f,
            )

# =========================================================
# CLI
# =========================================================

def parse_cli_arguments():

    parser = argparse.ArgumentParser(
        description="YOLO Dataset Splitter CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Path to images directory",
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels directory",
    )

    parser.add_argument(
        "--yaml",
        required=True,
        help="Path to dataset yaml file",
    )

    parser.add_argument(
        "--output",
        default="splits",
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    # -----------------------------------------------------
    # STANDARD SPLIT
    # -----------------------------------------------------

    split_parser = subparsers.add_parser(
        "split",
        help="Standard train/val/test split",
    )

    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )

    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )

    # -----------------------------------------------------
    # KFOLD
    # -----------------------------------------------------

    kfold_parser = subparsers.add_parser(
        "kfold",
        help="K-Fold dataset split",
    )

    kfold_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds",
    )
    
    # -----------------------------------------------------
    # STRATIFIED KFOLD
    # -----------------------------------------------------

    strat_kfold_parser = subparsers.add_parser(
        "stratified-kfold",
        help="Stratified K-Fold dataset split",
    )

    strat_kfold_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds",
    )

    return parser.parse_args()




if __name__ == "__main__":

    args = parse_cli_arguments()

    splitter = YOLODatasetSplitter(
        images_dir=args.images,
        labels_dir=args.labels,
        yaml_file=args.yaml,
        output_dir=args.output,
        seed=args.seed,
    )

    if args.mode == "split":

        splitter.split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    elif args.mode == "kfold":

        splitter.kfold_split(
            k=args.k,
        )
    
    elif args.mode == "stratified-kfold":

        splitter.stratified_kfold_split(
            k=args.k,
        )