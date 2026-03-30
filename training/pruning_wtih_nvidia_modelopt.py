from __future__ import annotations

import math
import argparse
from collections import OrderedDict

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA

import modelopt.torch.prune as mtp


# ================================
# Custom Trainer with Pruning
# ================================

class PrunedTrainer:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_train(self):
        """Setup training with FastNAS pruning."""
        super()._setup_train()

        # ----------------------------
        # Data collection function
        # ----------------------------

        def collect_func(batch):
            return self.preprocess_batch(batch)["img"]

        # ----------------------------
        # Model scoring function
        # ----------------------------

        def score_func(model):
            model.eval()

            # Disable logging temporarily
            self.validator.args.save = False
            self.validator.args.plots = False
            self.validator.args.verbose = False
            self.validator.args.data = "coco128.yaml"

            metrics = self.validator(model=model)

            # Restore settings
            self.validator.args.save = self.args.save
            self.validator.args.plots = self.args.plots
            self.validator.args.verbose = self.args.verbose
            self.validator.args.data = self.args.data

            return metrics["fitness"]

        # ----------------------------
        # Pruning constraint
        # ----------------------------

        prune_constraints = {
            "flops": "66%"
        }

        # Disable fusing requirement
        self.model.is_fused = lambda: True

        LOGGER.info("Starting FastNAS pruning...")

        self.model, prune_res = mtp.prune(
            model=self.model,
            mode="fastnas",
            constraints=prune_constraints,
            dummy_input=torch.randn(
                1,
                3,
                self.args.imgsz,
                self.args.imgsz
            ).to(self.device),
            config={
                "score_func": score_func,
                "checkpoint": "modelopt_fastnas_search_checkpoint.pth",
                "data_loader": self.train_loader,
                "collect_func": collect_func,
                "max_iter_data_loader": 20,
            },
        )

        LOGGER.info("Pruning completed")

        # ----------------------------
        # Move model to device
        # ----------------------------

        self.model.to(self.device)

        # ----------------------------
        # Recreate EMA
        # ----------------------------

        self.ema = ModelEMA(self.model)

        # ----------------------------
        # Rebuild optimizer
        # ----------------------------

        weight_decay = (
            self.args.weight_decay
            * self.batch_size
            * self.accumulate
            / self.args.nbs
        )

        iterations = math.ceil(
            len(self.train_loader.dataset)
            / max(self.batch_size, self.args.nbs)
        ) * self.epochs

        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        self._setup_scheduler()

        LOGGER.info("Optimizer and scheduler rebuilt after pruning")


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO with FastNAS pruning"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Model path"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset YAML"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/prune",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
    )

    return parser.parse_args()



# Main
def main():

    args = parse_args()

    LOGGER.info("Loading model...")

    model = YOLO(args.model)

    LOGGER.info("Starting training with pruning...")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        trainer=PrunedTrainer,
        exist_ok=True,
        warmup_epochs=0,
        project=args.project,
        name=args.name,
    )

    LOGGER.info("Training finished")


if __name__ == "__main__":
    main()