This repository contains useful scripts related to calculation and evaluation for ultralytics model:

- [YOLO K-Fold Evaluation](#YOLO-K-Fold-Evaluation)


# YOLO K-Fold Evaluation

## Model Directory Structure

Expected trained model structure:

```text
runs/
   synthetic_kfold/
       fold_1/
           weights/
               best.pt

       fold_2/
           weights/
               best.pt

       fold_3/
       fold_4/
       fold_5/
```

Each fold must contain:

```text
weights/best.pt
```

## Dataset Format

Expected YOLO dataset structure:

```text
dataset/
│
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
│
├── labels/
│   ├── img_001.txt
│   ├── img_002.txt
│   └── ...
│
└── dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: dataset

train: images
val: images
test: images

names:
  0: gun
```

## Basic Usage

```bash
python evaluate_kfold.py \
    --models runs/synthetic_kfold \
    --data dataset/test_black_gun/data.yaml \
    --output result/synthetic_kfold
```

## Example with Custom Parameters

```bash
python evaluate_kfold.py \
    --models runs/synthetic_kfold \
    --data dataset/test_black_gun/data.yaml \
    --output result/synthetic_kfold \
    --imgsz 1088 \
    --batch 16 \
    --conf 0.25 \
    --iou 0.7 \
    --device 0
```


## CLI Arguments

| Argument | Description                     |
| -------- | ------------------------------- |
| --models | Path to k-fold models directory |
| --data   | Dataset YAML file               |
| --output | Output directory                |
| --imgsz  | Image size                      |
| --batch  | Batch size                      |
| --conf   | Confidence threshold            |
| --iou    | IoU threshold                   |
| --device | CUDA device                     |

---

## Default Parameters

```text
Image Size: 640
Batch Size: 16
Confidence: 0.25
IoU: 0.7
Device: 0
```

