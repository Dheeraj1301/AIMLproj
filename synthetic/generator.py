"""Synthetic lithography process data generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LITHOGRAPHY_FEATURES: Final[list[str]] = [
    "etch_depth",
    "overlay_shift",
    "line_edge_roughness",
    "critical_dimension",
    "wafer_temperature",
    "mask_alignment_error",
    "exposure_dose",
    "photoresist_variation",
]

DEFECT_CLASSES: Final[list[str]] = [
    "nominal",
    "bridge_risk",
    "line_collapse",
    "overlay_escape",
    "dose_nonuniformity",
]


def generate_lithography_data(
    rows: int = 10_000,
    anomaly_rate: float = 0.06,
    drift_strength: float = 0.18,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Create a realistic-looking synthetic lithography dataset.

    The generator intentionally keeps the math simple while producing columns that
    resemble semiconductor process-control metrics. A smooth drift component and
    injected anomalies give downstream analytics enough signal for a strong demo.
    """

    rng = np.random.default_rng(random_state)
    time_index = np.arange(rows)
    drift = np.linspace(0, drift_strength, rows)
    lot_ids = rng.integers(1000, 1060, rows)
    wafer_ids = rng.integers(1, 26, rows)

    etch_depth = rng.normal(102.0, 3.5, rows) + drift * 8
    overlay_shift = rng.normal(0.0, 1.1, rows) + drift * 2.4
    line_edge_roughness = rng.normal(3.1, 0.38, rows) + drift * 1.1
    critical_dimension = rng.normal(45.0, 1.6, rows) - drift * 2.2
    wafer_temperature = rng.normal(21.4, 0.7, rows) + drift * 3.0
    mask_alignment_error = rng.normal(0.0, 0.45, rows) + overlay_shift * 0.18
    exposure_dose = rng.normal(24.0, 0.9, rows) + drift * 1.5
    photoresist_variation = rng.normal(1.0, 0.18, rows) + line_edge_roughness * 0.07

    data = pd.DataFrame(
        {
            "lot_id": lot_ids,
            "wafer_id": wafer_ids,
            "process_step": time_index,
            "etch_depth": etch_depth,
            "overlay_shift": overlay_shift,
            "line_edge_roughness": line_edge_roughness,
            "critical_dimension": critical_dimension,
            "wafer_temperature": wafer_temperature,
            "mask_alignment_error": mask_alignment_error,
            "exposure_dose": exposure_dose,
            "photoresist_variation": photoresist_variation,
        }
    )

    anomaly_count = max(1, int(rows * anomaly_rate))
    anomaly_idx = rng.choice(rows, anomaly_count, replace=False)
    anomaly_types = rng.choice(DEFECT_CLASSES[1:], anomaly_count)
    data["defect_class"] = "nominal"
    data.loc[anomaly_idx, "defect_class"] = anomaly_types

    # Controlled anomalies simulate dose, overlay, and morphology excursions.
    data.loc[anomaly_idx, "overlay_shift"] += rng.normal(4.5, 1.2, anomaly_count)
    data.loc[anomaly_idx, "line_edge_roughness"] += rng.normal(1.5, 0.45, anomaly_count)
    data.loc[anomaly_idx, "critical_dimension"] -= rng.normal(3.2, 0.9, anomaly_count)
    data.loc[anomaly_idx, "mask_alignment_error"] += rng.normal(1.7, 0.5, anomaly_count)

    defect_signal = (
        0.24 * np.abs(data["overlay_shift"])
        + 0.20 * data["line_edge_roughness"]
        + 0.18 * np.abs(data["critical_dimension"] - 45.0)
        + 0.14 * np.abs(data["mask_alignment_error"])
        + 0.07 * np.abs(data["wafer_temperature"] - 21.4)
    )
    defect_probability = 1 / (1 + np.exp(-(defect_signal - 2.4)))
    data["defect_probability"] = np.clip(defect_probability, 0, 1)
    data["defect_label"] = ((data["defect_probability"] > 0.55) | (data["defect_class"] != "nominal")).astype(int)
    data["yield_score"] = np.clip(100 - data["defect_probability"] * 55 - drift * 18, 0, 100)
    data["process_drift_index"] = np.round(drift * 100, 3)

    return data.round(4)


def save_synthetic_dataset(df: pd.DataFrame, output_dir: str | Path = "data") -> Path:
    """Save a generated dataset to disk and return the output path."""

    output_path = Path(output_dir) / "synthetic_lithography_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
