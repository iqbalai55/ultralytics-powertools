This repository contains useful scripts related to model training:

- [YOLO K-Fold Training](#YOLO-K-Fold-Training)


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
