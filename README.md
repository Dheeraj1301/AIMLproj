# LithoVision-AI

**Synthetic Lithography Analytics & Reconstruction Platform**

LithoVision-AI is a beginner-friendly but research-grade Streamlit application for semiconductor-style process analytics, synthetic wafer data generation, anomaly inference, and OpenCV-based image reconstruction.

## Features

- Upload lithography process CSV files.
- Clean missing values and duplicate rows automatically.
- Generate 10,000+ rows of synthetic wafer telemetry with drift, injected anomalies, and defect classes.
- Run Isolation Forest anomaly detection and RandomForest defect prediction when labels are available.
- Visualize statistical reports, correlations, PCA, process drift, and defect heatmaps.
- Upload wafer / SEM-like images for grayscale conversion, denoising, sharpening, reconstruction, and defect highlighting.
- Export processed CSV files, synthetic datasets, reconstructed images, and text reports.

## Project structure

```text
LithoVision_AI/
├── app.py
├── requirements.txt
├── analytics/
├── synthetic/
├── reconstruction/
├── utils/
├── assets/
├── data/
├── models/
└── outputs/
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Example columns

The analytics pipeline works best with semiconductor-style numeric columns such as:

- `etch_depth`
- `overlay_shift`
- `line_edge_roughness`
- `critical_dimension`
- `wafer_temperature`
- `mask_alignment_error`
- `exposure_dose`
- `photoresist_variation`

If no CSV is uploaded, the app automatically generates and analyzes a synthetic lithography dataset.
