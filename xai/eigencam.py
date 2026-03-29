import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def apply_letterbox_resize(image_bgr, target_size=640):
    original_height, original_width = image_bgr.shape[:2]

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    scale_ratio = min(
        target_size[1] / original_height,
        target_size[0] / original_width,
    )

    resized_width = int(original_width * scale_ratio)
    resized_height = int(original_height * scale_ratio)

    resized_image = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
    )

    pad_top = (target_size[1] - resized_height) // 2
    pad_bottom = target_size[1] - resized_height - pad_top
    pad_left = (target_size[0] - resized_width) // 2
    pad_right = target_size[0] - resized_width - pad_left

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded_image


def load_and_prepare_image(image_path, image_size=640):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_bgr = apply_letterbox_resize(
        image_bgr,
        image_size,
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_float = (
        image_rgb.astype(np.float32)
        / 255.0
    )

    return image_rgb, image_float


# =========================================================
# YOLO WRAPPER
# =========================================================


class YOLOEigenCAMForwardWrapper(torch.nn.Module):

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, input_tensor):
        output = self.yolo_model(input_tensor)

        while isinstance(output, (tuple, list)):
            output = output[0]

        return output


# =========================================================
# NORMALIZATION
# =========================================================


def normalize_eigencam_heatmap(heatmap):
    min_value = heatmap.min()
    max_value = heatmap.max()

    if max_value - min_value == 0:
        return heatmap

    return (heatmap - min_value) / (
        max_value - min_value
    )


# =========================================================
# SHARPENING
# =========================================================


def sharpen_eigencam_heatmap(heatmap):
    laplacian = cv2.Laplacian(
        heatmap,
        cv2.CV_32F,
    )

    sharpened = heatmap - 0.5 * laplacian

    return np.clip(
        sharpened,
        0,
        1,
    )


# =========================================================
# MULTI-LAYER EIGENCAM
# =========================================================


def compute_single_layer_eigencam(
    model,
    input_tensor,
    target_layer,
):

    eigencam_algorithm = EigenCAM(
        model=YOLOEigenCAMForwardWrapper(
            model.model
        ),
        target_layers=[target_layer],
    )

    with torch.no_grad():
        eigencam_heatmap = eigencam_algorithm(
            input_tensor=input_tensor,
            eigen_smooth=True,
        )[0]

    eigencam_heatmap = np.nan_to_num(
        eigencam_heatmap,
        nan=0.0,
    )

    return eigencam_heatmap


def compute_multi_layer_eigencam(
    model,
    input_tensor,
    target_layers,
    enable_multi_layer_merge=True,
    merge_weights=None,
    enable_normalization=True,
    enable_sharpening=False,
):

    heatmaps = []

    for layer in target_layers:
        heatmap = compute_single_layer_eigencam(
            model,
            input_tensor,
            layer,
        )

        heatmaps.append(heatmap)

    heatmaps = np.stack(
        heatmaps,
        axis=0,
    )

    if enable_multi_layer_merge:

        if merge_weights is None:
            merge_weights = (
                np.ones(len(target_layers))
                / len(target_layers)
            )

        merged_heatmap = np.sum(
            heatmaps * merge_weights[:, None, None],
            axis=0,
        )

    else:
        merged_heatmap = heatmaps[-1]

    if enable_normalization:
        merged_heatmap = normalize_eigencam_heatmap(
            merged_heatmap
        )

    if enable_sharpening:
        merged_heatmap = sharpen_eigencam_heatmap(
            merged_heatmap
        )

    return merged_heatmap


# =========================================================
# MAIN RUNNER
# =========================================================


def run_eigencam_from_cli(args):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    detection_model = YOLO(
        args.weights
    )

    detection_model.model.to(
        device
    ).eval()

    image_rgb, image_float = load_and_prepare_image(
        args.image,
        args.imgsz,
    )

    input_tensor = torch.from_numpy(
        image_float.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    target_layers = [
        detection_model.model.model[idx]
        for idx in args.layers
    ]

    merge_weights = None

    if args.merge_weights is not None:
        merge_weights = np.array(
            args.merge_weights,
            dtype=np.float32,
        )

    heatmap = compute_multi_layer_eigencam(
        detection_model,
        input_tensor,
        target_layers,
        enable_multi_layer_merge=args.enable_merge,
        merge_weights=merge_weights,
        enable_normalization=args.enable_normalization,
        enable_sharpening=args.enable_sharpening,
    )

    visualization = show_cam_on_image(
        image_float,
        heatmap,
        use_rgb=True,
    )

    output_path = args.output

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.axis("off")

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
    )

    print("Saved:", output_path)


# =========================================================
# CLI
# =========================================================


def parse_cli_arguments():

    parser = argparse.ArgumentParser(
        description="Multi-layer EigenCAM visualization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO weights",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices (example: 16 19 22)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eigencam_output.png",
        help="Output image path",
    )

    parser.add_argument(
        "--enable-merge",
        action="store_true",
        help="Enable multi-layer merge",
    )

    parser.add_argument(
        "--merge-weights",
        type=float,
        nargs="+",
        help="Merge weights (example: 0.3 0.4 0.3)",
    )

    parser.add_argument(
        "--enable-normalization",
        action="store_true",
        help="Enable heatmap normalization",
    )

    parser.add_argument(
        "--enable-sharpening",
        action="store_true",
        help="Enable Laplacian sharpening",
    )

    return parser.parse_args()


if __name__ == "__main__":

    cli_args = parse_cli_arguments()

    run_eigencam_from_cli(cli_args)
