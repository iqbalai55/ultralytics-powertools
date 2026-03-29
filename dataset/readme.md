This repository contains useful scripts related to dataset preparation and management:

- [YOLO Dataset Splitter](#yolo-dataset-splitter)
- [LabelMe to YOLO Conversion](#labelme-to-yolo-conversion)
- [Visualization of YOLO Annotations](#visualize-yolo-annotations)


# Acceptable Dataset Format

Expected YOLO structure:

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

# YOLO Dataset Splitter
## Basic Usage

### Standard Train / Val / Test Split

```bash
python split_dataset.py \
    --images dataset/images \
    --labels dataset/labels \
    --yaml dataset.yaml \
    split
```

Default ratios:

```text
Train: 70%
Validation: 20%
Test: 10%
```

Custom ratios:

```bash
python split_dataset.py \
    --images dataset/images \
    --labels dataset/labels \
    --yaml dataset.yaml \
    split \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

## K-Fold Split

```bash
python split_dataset.py \
    --images dataset/images \
    --labels dataset/labels \
    --yaml dataset.yaml \
    kfold \
    --k 5
```

## Stratified K-Fold Split

Stratified K-Fold maintains class distribution across folds using the dominant class per image.

```bash
python split_dataset.py \
    --images dataset/images \
    --labels dataset/labels \
    --yaml dataset.yaml \
    stratified-kfold \
    --k 5
```


## Output Structure

### Standard Split

```text
splits/
   2026-XX-XX_standard/
       train/
           images/
           labels/
       val/
           images/
           labels/
       test/
           images/
           labels/
       dataset.yaml
```

### K-Fold / Stratified K-Fold

```text
splits/
   2026-XX-XX_kfold/
       fold_1/
           train/
           val/
           fold_1.yaml

       fold_2/
       fold_3/
       fold_4/
       fold_5/
```

Each fold contains its own dataset configuration file ready for training.

## CLI Arguments

| Argument | Description              |
| -------- | ------------------------ |
| --images | Path to images directory |
| --labels | Path to labels directory |
| --yaml   | Dataset YAML file        |
| --output | Output directory         |
| --seed   | Random seed              |

Modes:

| Mode             | Description              |
| ---------------- | ------------------------ |
| split            | Train / Val / Test split |
| kfold            | K-Fold split             |
| stratified-kfold | Stratified K-Fold split  |

Add the following section **above `# Dataset Format`** (or at the very top of the README) to guide users on converting LabelMe annotations before splitting.
Below is **only the section you need to add**, consistent with your current README style.


# LabelMe to YOLO Conversion

## Expected LabelMe Dataset

```text
raw_data/
│
├── image_001.jpg
├── image_001.json
├── image_002.jpg
├── image_002.json
└── ...
```

Each JSON must contain rectangle annotations.

---

## Convert LabelMe to YOLO

```bash
python labelme_to_yolo.py \
    --input raw_data/test_video \
    --output dataset/test_black_gun
```


Add the following section to your README to document the visualization script.
Place it **after the LabelMe conversion section** or **after Dataset Format**.
Below is **only the content to append** to your existing README.


# Visualize YOLO Annotations

## Basic Usage

```bash
python visualize_yolo.py \
    --input dataset/test_black_gun \
    --output viz_dir
```

## Output Structure

```text
viz_dir/
│
├── img_001.jpg
├── img_002.jpg
└── ...
```

Bounding boxes will be drawn on each image.




## CLI Arguments

| Argument | Description                    |
| -------- | ------------------------------ |
| --input  | Dataset directory              |
| --output | Visualization output directory |

Default:

```text
Output directory: viz_output
```
