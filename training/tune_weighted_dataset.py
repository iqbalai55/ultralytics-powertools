from __future__ import annotations

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

        print("\nClass counts:", self.counts)
        print("Class weights:", self.class_weights)

    # ----------------------------

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]

        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for idx in cls:
                self.counts[idx] += 1

        self.counts = np.array(self.counts)

        # Avoid division by zero
        self.counts = np.where(self.counts == 0, 1, self.counts)

    # ----------------------------

    def calculate_weights(self):
        weights = []

        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1)
                continue

            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)

        return weights

    # ----------------------------

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    # ----------------------------

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))

        index = np.random.choice(len(self.labels), p=self.probabilities)

        return self.transforms(self.get_image_and_label(index))


# ================================
# Monkey Patch Dataset
# ================================

build.YOLODataset = YOLOWeightedDataset


# ================================
# MAIN (NO CLI + TUNING)
# ================================

if __name__ == "__main__":

    print("\n🔥 Start YOLO Hyperparameter Tuning with Weighted Sampling\n")

    # ====================
    # CONFIG (EDIT HERE)
    # ====================

    DATA = "dataset/indrapuri_yolo/data.yaml"
    MODEL = "yolov8m.pt"

    EPOCHS = 500
    ITERATIONS = 50   # jumlah trial tuning
    DEVICE = "0"

    # ====================
    # SEARCH SPACE
    # ====================

    search_space = {
        "lr0": (1e-6, 1e-2),
        "lrf": (0.001, 0.3),
        "momentum": (0.6, 0.98),
    }

    # ====================
    # MODEL
    # ====================

    model = YOLO(MODEL)

    # ====================
    # TUNING
    # ====================

    results = model.tune(
        data=DATA,
        epochs=EPOCHS,
        iterations=ITERATIONS,
        device=DEVICE,
        space=search_space,
        project="runs/yolov8m_tune_weighted",
        name="exp",
        exist_ok=True,
        seed=42,
    )

    print("\n✅ Tuning completed")