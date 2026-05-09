"""Reporting helpers for LithoVision-AI."""

from __future__ import annotations

import pandas as pd


def build_text_report(processed: pd.DataFrame, insights: list[str], drift_status: str) -> str:
    """Build a lightweight exportable analytics report."""

    lines = [
        "LithoVision-AI Synthetic Lithography Analytics Report",
        "======================================================",
        f"Rows analyzed: {len(processed):,}",
        f"Average wafer quality score: {processed['wafer_quality_score'].mean():.2f}",
        f"Average defect probability: {processed['defect_probability'].mean():.4f}",
        f"Anomaly count: {int(processed['anomaly_flag'].sum())}",
        f"Process drift status: {drift_status}",
        "",
        "AI Insights:",
    ]
    lines.extend(f"- {insight}" for insight in insights)
    return "\n".join(lines)
