from pathlib import Path
import argparse

import cv2
from tqdm import tqdm


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_class_mapping(dataset_dir: Path):
    """
    Load classes.txt mapping.

    Returns:
        dict[int, str]
    """

    classes_file = dataset_dir / "classes.txt"

    if not classes_file.exists():
        print("Warning: classes.txt not found")
        return {}

    with open(classes_file, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    return {idx: label for idx, label in enumerate(labels)}


def yolo_to_pixel_bbox(xc, yc, bw, bh, width, height):
    """
    Convert normalized YOLO bbox to pixel coordinates.
    """

    xc *= width
    yc *= height
    bw *= width
    bh *= height

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    return x1, y1, x2, y2


def draw_boxes(
    image,
    label_file: Path,
    class_map,
):
    """
    Draw bounding boxes on image.
    """

    height, width = image.shape[:2]

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, xc, yc, bw, bh = map(float, parts)

        x1, y1, x2, y2 = yolo_to_pixel_bbox(
            xc,
            yc,
            bw,
            bh,
            width,
            height,
        )

        label_text = class_map.get(
            int(class_id),
            str(int(class_id)),
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            label_text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def visualize_yolo(
    input_dir: str,
    output_dir: str,
):
    """
    Visualize YOLO bounding boxes.

    Expected structure:

        dataset/
            images/
            labels/
            classes.txt
    """

    dataset_path = Path(input_dir)
    output_path = Path(output_dir)

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_map = load_class_mapping(dataset_path)

    image_files = sorted(
        f for f in images_dir.glob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"Found {len(image_files)} images")

    saved = 0

    for img_file in tqdm(
        image_files,
        desc="Visualizing",
    ):

        image = cv2.imread(str(img_file))

        if image is None:
            print(f"Warning: cannot read {img_file}")
            continue

        label_file = labels_dir / f"{img_file.stem}.txt"

        if label_file.exists():
            draw_boxes(
                image,
                label_file,
                class_map,
            )

        save_path = output_path / img_file.name

        cv2.imwrite(
            str(save_path),
            image,
        )

        saved += 1

    print()
    print("Visualization complete")
    print("Saved images:", saved)
    print("Output directory:", output_path)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Dataset directory",
    )

    parser.add_argument(
        "--output",
        default="viz_output",
        help="Output visualization directory",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    visualize_yolo(
        input_dir=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()