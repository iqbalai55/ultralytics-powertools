This repository contains useful scripts related to model explainability:

- [Multi-Layer EigenCAM ](#Multi-Layer-EigenCAM)

# Multi-Layer EigenCAM 

## Basic Command

Run EigenCAM on a single model and image.

```bash
python eigencam.py \
    --image dataset/test_black_gun/images/frame_00010.png \
    --weights runs/real_kfold/fold_4/weights/best.pt \
    --layers 16 19 22
```

If successful, the script will print:

```text
Saved: eigencam_output.png
```

And generate:

```text
eigencam_output.png
```

## Enable Multi-Layer Merge

```bash
python eigencam.py \
    --image frame.png \
    --weights best.pt \
    --layers 16 19 22 \
    --enable-merge
```


##  Custom Merge Weights

Example: P3 / P4 / P5 weighting.

```bash
python eigencam.py \
    --image frame.png \
    --weights best.pt \
    --layers 16 19 22 \
    --enable-merge \
    --merge-weights 0.3 0.4 0.3
```

## Full Example (All Options)

```bash
python eigencam.py \
    --image dataset/test_black_gun/images/frame_00010.png \
    --weights runs/real_kfold/fold_4/weights/best.pt \
    --layers 16 19 22 \
    --imgsz 1088 \
    --enable-merge \
    --merge-weights 0.3 0.4 0.3 \
    --enable-normalization \
    --enable-sharpening \
    --output result_cam.png
```

## CLI Arguments Reference

| Argument                 | Description                |
| ------------------------ | -------------------------- |
| `--image`                | Path to input image        |
| `--weights`              | Path to YOLO model weights |
| `--layers`               | Target layer indices       |
| `--imgsz`                | Inference image size       |
| `--enable-merge`         | Enable multi-layer fusion  |
| `--merge-weights`        | Weights for layer fusion   |
| `--enable-normalization` | Normalize heatmap          |
| `--enable-sharpening`    | Apply Laplacian sharpening |
| `--output`               | Output file path           |

