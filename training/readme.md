This repository contains useful scripts related to model training:

- [YOLO K-Fold Training](#YOLO-K-Fold-Training)
- [Pruning with NVIDIA ModelOpt](#Pruning-with-NVIDIA-ModelOpt)
- [Training with weighted dataset](#Training-with-weighted-dataset)


# YOLO K-Fold Training

## K-Fold Dataset Structure

Expected directory structure:

```text
kfold_dataset/
   2026-03-25_kfold/

       fold_1/
           train/
               images/
               labels/

           val/
               images/
               labels/

           fold_1.yaml

       fold_2/
           fold_2.yaml

       fold_3/
       fold_4/
       fold_5/
```

Each fold must contain:

```text
fold_X.yaml
```

Example:

```yaml
path: fold_1

train: train/images
val: val/images

names:
  0: gun
```

## Basic Usage

```bash
python train_kfold.py \
    --kfold-dir dataset/kfold/2026-03-25_kfold
```

This will:

```text
Train all folds sequentially
Save models into runs/kfold/
```


## Example with Custom Parameters

```bash
python train_kfold.py \
    --kfold-dir dataset/kfold/2026-03-25_kfold \
    --weights yolo26m.pt \
    --epochs 30 \
    --batch 8 \
    --imgsz 640 \
    --device 0 \
    --project runs/combined_kfold
```


## Resume Training from Specific Fold

Useful if training was interrupted.

```bash
python train_kfold.py \
    --kfold-dir dataset/kfold \
    --start-fold 3
```

This will:

```text
Skip Fold 1
Skip Fold 2
Train Fold 3 onward
```


## Default Training Settings

```text
Weights: yolo26m.pt
Epochs: 100
Batch Size: 16
Image Size: 640
Device: 0
Seed: 42
Workers: 8
```

Augmentations:

```text
HSV Hue: 0.02
HSV Saturation: 0.9
HSV Value: 0.8
Scale: 0.3
Shear: 10.0
Random Erasing: 0.3
```


## CLI Arguments

| Argument          | Description                      |
| ----------------- | -------------------------------- |
| --kfold-dir       | Path to K-Fold dataset directory |
| --weights         | Base model weights               |
| --epochs          | Number of training epochs        |
| --batch           | Batch size                       |
| --imgsz           | Image size                       |
| --device          | GPU device                       |
| --project         | Output directory                 |
| --run-name-prefix | Fold name prefix                 |
| --start-fold      | Resume from fold number          |


## Pruning with NVIDIA ModelOpt

This training script supports **automatic structured pruning** using **NVIDIA ModelOpt (FastNAS)** before training begins.

The model is pruned based on **FLOPs constraints**, then training continues normally using the pruned architecture.


## Basic Usage

```bash
python train_pruned_yolo.py \
    --model yolov8m.pt \
    --data dataset/data.yaml
```

## Example with Custom Parameters

```bash
python train_pruned_yolo.py \
    --model yolov8m.pt \
    --data dataset/data.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/pruned_train \
    --name exp_pruned
```


## Pruning Configuration

Default pruning constraint:

```text
Pruning Mode: fastnas
Target FLOPs: 66%
Max Iterations: 20
Checkpoint: modelopt_fastnas_search_checkpoint.pth
```

From code:

```python
prune_constraints = {
    "flops": "66%"
}
```

Meaning:

```text
The model will be pruned to approximately 66% of the original FLOPs
```

Here is a **single-file CLI training script** for your **Weighted Dataset sampling** implementation.
It keeps your logic intact, makes it reusable, and exposes configurable parameters like **model**, **dataset**, **epochs**, etc.

This is structured like a typical production training utility (similar style to your k-fold and pruning scripts).

# Training with weighted dataset



## Basic

```bash
python train_weighted_yolo.py \
    --data dataset/data.yaml
```


## Custom Model

```bash
python train_weighted_yolo.py \
    --data dataset/data.yaml \
    --model yolo26m.pt
```


## Full Training Example

```bash
python train_weighted_yolo.py \
    --data dataset/data.yaml \
    --model yolo26m.pt \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --device 0 \
    --project runs/traffic_weighted
```

## CLI Arguments

| Argument  | Description                  |
| --------- | ---------------------------- |
| --data    | Path to dataset YAML file    |
| --model   | Base model weights           |
| --epochs  | Number of training epochs    |
| --batch   | Batch size                   |
| --imgsz   | Image size                   |
| --device  | GPU device                   |
| --workers | Number of dataloader workers |
| --project | Output directory             |
| --name    | Run name                     |
| --seed    | Random seed                  |


## Default Training Settings

```text
Model: yolov8n.pt
Epochs: 50
Batch Size: 16
Image Size: 640
Device: 0
Workers: 8
Seed: 42
```
