from __future__ import annotations

import argparse
import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build


# ================================
# Weighted Dataset
# ================================

class YOLOWeightedDataset(YOLODataset):
    """
    Weighted sampling dataset for handling class imbalance.
    """

    def __init__(self, *args, mode="train", **kwargs):

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # Count class instances
        self.count_instances()

        # Inverse frequency weighting
        class_weights = np.sum(self.counts) / self.counts

        self.class_weights = np.array(class_weights)

        # Aggregation function
        self.agg_func = np.mean

        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

        print("\nClass counts:")
        print(self.counts)

        print("\nClass weights:")
        print(self.class_weights)

    # ----------------------------

    def count_instances(self):

        self.counts = [0 for _ in range(len(self.data["names"]))]

        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            for idx in cls:
                self.counts[idx] += 1

        self.counts = np.array(self.counts)

        # Avoid division by zero
        self.counts = np.where(
            self.counts == 0,
            1,
            self.counts,
        )

    # ----------------------------

    def calculate_weights(self):

        weights = []

        for label in self.labels:

            cls = label["cls"].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1)
                continue

            weight = self.agg_func(
                self.class_weights[cls]
            )

            weights.append(weight)

        return weights

    # ----------------------------

    def calculate_probabilities(self):

        total_weight = sum(self.weights)

        probabilities = [
            w / total_weight
            for w in self.weights
        ]

        return probabilities

    # ----------------------------

    def __getitem__(self, index):

        if not self.train_mode:
            return self.transforms(
                self.get_image_and_label(index)
            )

        index = np.random.choice(
            len(self.labels),
            p=self.probabilities,
        )

        return self.transforms(
            self.get_image_and_label(index)
        )


# ================================
# Monkey Patch Dataset
# ================================

build.YOLODataset = YOLOWeightedDataset


# ================================
# CLI Arguments
# ================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train YOLO with Weighted Dataset Sampling"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset YAML file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model weights",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/weighted_train",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ================================
# Main
# ================================

def main():

    args = parse_args()

    print("\nStarting training with weighted sampling\n")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
    )

    print("\nTraining completed")


if __name__ == "__main__":
    main()