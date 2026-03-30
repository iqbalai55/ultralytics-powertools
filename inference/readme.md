
This repository contains useful scripts related to **model inference** using Ultralytics YOLO models.
The scripts support running inference on images, videos, folders, streams, and exporting output results.

* [Tracking With Reid](#Tracking-With-Reid)


# Tracking With Reid


## Supported Input Sources

The inference script supports multiple input types:

```text
Image file:
    image.jpg

Video file:
    video.mp4

Folder:
    dataset/images/

Webcam:
    0

RTSP stream:
    rtsp://user:pass@ip:554/stream

URL:
    https://example.com/video.mp4
```

---

## Basic Usage

```bash
python track.py \
    --model best.pt \
    --source video.mp4 \
    --save
```

## Example with Custom Parameters

```bash
python track.py \
    --model runs/train/exp/weights/best.pt \
    --source dataset/images \
    --tracker botsort.yaml \
    --imgsz 1088 \
    --conf 0.25 \
    --iou 0.7 \
    --device 0 \
    --persist \
    --save
```


## Example: Webcam Inference

```bash
python track.py \
    --model best.pt \
    --source 0 \
    --show
```


## Example: Save to Custom Output File

```bash
python track.py \
    --model best.pt \
    --source video.mp4 \
    --save \
    --output result/output_video.mp4
```


## CLI Arguments

| Argument  | Description                                         |
| --------- | --------------------------------------------------- |
| --model   | Path to YOLO model                                  |
| --source  | Input source (image, video, folder, webcam, stream) |
| --tracker | Tracker configuration file                          |
| --imgsz   | Image size                                          |
| --conf    | Confidence threshold                                |
| --iou     | IoU threshold                                       |
| --device  | CUDA device                                         |
| --persist | Maintain tracking IDs across frames                 |
| --save    | Save output results                                 |
| --show    | Display results in window                           |
| --project | Output project directory                            |
| --name    | Experiment name                                     |
| --output  | Custom output file path                             |

---

## Default Parameters

```text
Image Size: 640
Confidence: 0.25
IoU: 0.7
Tracker: botsort.yaml
Device: 0
Persist: False
Save: False
```

