from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# Embedded BoT-SORT YAML
# --------------------------------------------------

BOTSORT_YAML = """
# Ultralytics 🚀 AGPL-3.0 License

tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25

track_buffer: 15
match_thresh: 0.8
fuse_score: True

gmc_method: sparseOptFlow

proximity_thresh: 0.5
appearance_thresh: 0.8

with_reid: True
model: auto
"""


def ensure_tracker_file(path: str) -> str:
    """
    Ensure tracker YAML exists.
    If not, create it from embedded config.
    """

    tracker_path = Path(path)

    if not tracker_path.exists():
        tracker_path.write_text(BOTSORT_YAML.strip())

    return str(tracker_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal YOLO Tracking CLI with output return"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (e.g. yolo26n.pt)",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (video, webcam, RTSP, URL, folder)",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        help="Tracker config",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--persist",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/track",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output file path",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------
    # Ensure tracker YAML exists
    # --------------------------------------------------

    args.tracker = ensure_tracker_file(args.tracker)

    model = YOLO(args.model)

    results = model.track(
        source=args.source,
        tracker=args.tracker,
        device=args.device,
        persist=args.persist,
        save=args.save,
        project=args.project,
        name=args.name,
    )

    # Resolve output directory
    save_dir = None

    if results:
        save_dir = Path(results[0].save_dir)

    # Find output file
    output_file = None

    if save_dir and save_dir.exists():
        videos = (
            list(save_dir.glob("*.mp4"))
            + list(save_dir.glob("*.avi"))
        )

        if videos:
            output_file = videos[0]

    # Optional: move to custom path
    if args.output and output_file:
        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file.rename(output_path)
        output_file = output_path

    # Print output
    if output_file:
        print(f"OUTPUT_FILE={output_file.resolve()}")
    else:
        print("No output file generated.")


if __name__ == "__main__":
    main()