"""OpenCV-based lithography image reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class ReconstructionResult:
    """Outputs from the image reconstruction workflow."""

    original: np.ndarray
    denoised: np.ndarray
    reconstructed: np.ndarray
    defect_heatmap: np.ndarray
    highlighted: np.ndarray
    metrics: dict[str, float]
    output_path: Path


def pil_to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image to an 8-bit grayscale OpenCV array."""

    return np.array(image.convert("L"))


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _similarity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = np.mean(np.abs(original.astype("float32") - reconstructed.astype("float32")))
    return float(np.clip(1 - diff / 255.0, 0, 1))


def reconstruct_lithography_image(
    image: Image.Image,
    output_dir: str | Path = "outputs",
    filename: str = "reconstructed_lithography.png",
) -> ReconstructionResult:
    """Denoise, sharpen, highlight defects, and save the reconstructed image."""

    original = pil_to_grayscale(image)
    denoised = cv2.fastNlMeansDenoising(original, None, h=12, templateWindowSize=7, searchWindowSize=21)
    gaussian = cv2.GaussianBlur(denoised, (3, 3), 0)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5.4, -1], [0, -1, 0]])
    reconstructed = cv2.filter2D(gaussian, -1, sharpen_kernel)
    reconstructed = cv2.equalizeHist(reconstructed)

    edges = cv2.Canny(reconstructed, 60, 150)
    residual = cv2.absdiff(original, reconstructed)
    _, defects = cv2.threshold(residual, int(np.percentile(residual, 92)), 255, cv2.THRESH_BINARY)
    defect_heatmap = cv2.applyColorMap(cv2.addWeighted(edges, 0.45, defects, 0.75, 0), cv2.COLORMAP_INFERNO)

    highlighted = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
    highlighted[defects > 0] = [45, 45, 255]
    highlighted = cv2.addWeighted(highlighted, 0.72, defect_heatmap, 0.28, 0)

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), reconstructed)

    metrics = {
        "original_sharpness": round(_sharpness(original), 3),
        "reconstructed_sharpness": round(_sharpness(reconstructed), 3),
        "similarity_score": round(_similarity(original, reconstructed), 4),
        "defect_area_percent": round(float((defects > 0).mean() * 100), 3),
        "ai_reconstruction_confidence": round(float(np.clip(_similarity(original, reconstructed) * 0.74 + 0.18, 0, 0.99)), 4),
    }

    return ReconstructionResult(
        original=original,
        denoised=denoised,
        reconstructed=reconstructed,
        defect_heatmap=defect_heatmap,
        highlighted=highlighted,
        metrics=metrics,
        output_path=output_path,
    )
