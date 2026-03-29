import json
import shutil
import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image


def discover_labels(input_dir: Path):
    """
    Discover all labels from LabelMe JSON files.
    Returns deterministic sorted label list.
    """

    labels = set()

    for json_file in input_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label:
                labels.add(label)

    return sorted(labels)


def build_label_mapping(labels):
    """
    Create label -> id mapping.
    """

    return {label: idx for idx, label in enumerate(labels)}


def find_image(json_file: Path, input_dir: Path, image_exts):
    """
    Find corresponding image for JSON.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    image_name = data.get("imagePath")

    if image_name:
        image_path = input_dir / image_name
        if image_path.exists():
            return image_path

    for ext in image_exts:
        candidate = input_dir / f"{json_file.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def get_image_size(data, image_path):
    """
    Get image width and height.
    """

    if "imageWidth" in data and "imageHeight" in data:
        return data["imageWidth"], data["imageHeight"]

    with Image.open(image_path) as img:
        return img.size


def convert_shape_to_yolo(points, width, height):
    """
    Convert rectangle points to YOLO bbox.
    """

    if len(points) < 2:
        return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    return (
        x_center,
        y_center,
        bbox_width,
        bbox_height,
    )


def save_classes_file(labels, output_dir: Path):
    """
    Save classes.txt
    """

    classes_file = output_dir / "classes.txt"

    with open(classes_file, "w") as f:
        for label in labels:
            f.write(label + "\n")


def labelme_to_yolo(
    input_dir: str,
    output_dir: str,
    image_exts=(".jpg", ".jpeg", ".png"),
):
    """
    Convert LabelMe dataset to YOLO format.

    Output structure:

        output/
            images/
            labels/
            classes.txt
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    json_files = list(input_path.glob("*.json"))

    if not json_files:
        raise RuntimeError("No JSON files found")

    print(f"Found {len(json_files)} JSON files")

    # Discover labels automatically
    labels = discover_labels(input_path)
    label_to_id = build_label_mapping(labels)

    print("Detected labels:")
    for label, idx in label_to_id.items():
        print(f"{idx}: {label}")

    save_classes_file(labels, output_path)

    converted = 0

    for json_file in tqdm(json_files, desc="Converting"):

        with open(json_file, "r") as f:
            data = json.load(f)

        image_path = find_image(
            json_file,
            input_path,
            image_exts,
        )

        if image_path is None:
            print(f"Image not found for {json_file.name}")
            continue

        shutil.copy2(
            image_path,
            images_dir / image_path.name,
        )

        width, height = get_image_size(
            data,
            image_path,
        )

        yolo_lines = []

        for shape in data.get("shapes", []):

            label = shape.get("label")

            if label not in label_to_id:
                continue

            bbox = convert_shape_to_yolo(
                shape.get("points", []),
                width,
                height,
            )

            if bbox is None:
                continue

            class_id = label_to_id[label]

            x, y, w, h = bbox

            line = (
                f"{class_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

            yolo_lines.append(line)

        label_file = labels_dir / f"{json_file.stem}.txt"

        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print()
    print("Conversion complete")
    print("Images:", images_dir)
    print("Labels:", labels_dir)
    print("Classes:", output_path / "classes.txt")
    print("Converted files:", converted)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Convert LabelMe annotations to YOLO format"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to LabelMe JSON directory"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset directory"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    labelme_to_yolo(
        input_dir=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()