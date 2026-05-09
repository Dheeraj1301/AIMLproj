"""LithoVision-AI Streamlit application."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from analytics.csv_analytics import analyze_csv
from reconstruction.image_pipeline import reconstruct_lithography_image
from synthetic.generator import generate_lithography_data, save_synthetic_dataset
from utils.reporting import build_text_report

st.set_page_config(
    page_title="LithoVision-AI",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%); color: #e5e7eb;}
    section[data-testid="stSidebar"] {background: #020617; border-right: 1px solid #1e3a8a;}
    .metric-card {padding: 1rem; border: 1px solid #2563eb; border-radius: 18px; background: rgba(15, 23, 42, 0.86);}
    .hero {padding: 1.2rem 1.4rem; border-radius: 22px; background: linear-gradient(90deg, rgba(37,99,235,.28), rgba(20,184,166,.20)); border: 1px solid rgba(96,165,250,.38);}
    .small-label {font-size: .85rem; color: #93c5fd; text-transform: uppercase; letter-spacing: .08em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="small-label">{label}</div>
          <h2 style="margin:.2rem 0;color:#f8fafc;">{value}</h2>
          <p style="margin:0;color:#cbd5e1;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


with st.sidebar:
    st.title("🧫 LithoVision-AI")
    st.caption("Synthetic Lithography Analytics & Reconstruction Platform")
    inference_mode = st.selectbox(
        "Inference mode",
        ["Research Demo", "Fast Screening", "Yield Optimization", "Defect Localization"],
    )
    synthetic_rows = st.slider("Synthetic rows", 1_000, 50_000, 10_000, 1_000)
    anomaly_rate = st.slider("Anomaly rate", 0.01, 0.20, 0.06, 0.01)
    generate_button = st.button("Generate synthetic wafer data", use_container_width=True)
    st.divider()
    uploaded_csv = st.file_uploader("Upload lithography CSV", type=["csv"])
    uploaded_image = st.file_uploader("Upload wafer / SEM-like image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if generate_button or "synthetic_df" not in st.session_state:
    with st.spinner("Generating synthetic process telemetry with controlled drift..."):
        st.session_state.synthetic_df = generate_lithography_data(rows=synthetic_rows, anomaly_rate=anomaly_rate)
        st.session_state.synthetic_path = save_synthetic_dataset(st.session_state.synthetic_df)

st.markdown(
    """
    <div class="hero">
      <div class="small-label">Computational lithography intelligence</div>
      <h1 style="margin:.2rem 0 0 0;">LithoVision-AI</h1>
      <p style="font-size:1.05rem;color:#dbeafe;max-width:980px;">
        A research-grade dashboard for synthetic wafer generation, process-aware anomaly inference,
        yield analytics, and OpenCV-based lithography image reconstruction.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

source_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else st.session_state.synthetic_df
analysis = analyze_csv(source_df)

tabs = st.tabs([
    "Dashboard Overview",
    "CSV Analytics",
    "Synthetic Data Generator",
    "Image Reconstruction",
    "AI Insights",
])

with tabs[0]:
    st.subheader("Semiconductor Process Intelligence Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Wafer Quality Score", f"{analysis.yield_score:.1f}/100", "Composite yield optimization score")
    with col2:
        render_metric_card("Defect Probability", f"{analysis.defect_probability:.2%}", "Average AI defect-risk inference")
    with col3:
        render_metric_card("Anomaly Records", f"{int(analysis.processed['anomaly_flag'].sum()):,}", "Isolation Forest process excursions")
    with col4:
        accuracy_text = f"{analysis.model_accuracy:.1%}" if analysis.model_accuracy is not None else "Heuristic"
        render_metric_card("RF Defect Model", accuracy_text, "RandomForest validation accuracy")

    st.progress(min(max(analysis.yield_score / 100, 0), 1), text="Yield optimization readiness")
    st.info(analysis.drift_status)

    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = px.line(
            analysis.processed.reset_index(),
            x="index",
            y="wafer_quality_score",
            title="Wafer Quality Score Across Process Sequence",
            template="plotly_dark",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            analysis.processed,
            x="defect_probability",
            color="anomaly_flag",
            nbins=35,
            title="Defect Probability Distribution",
            template="plotly_dark",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("CSV Analytics Workbench")
    st.caption("Upload process CSV files or use the generated wafer telemetry dataset.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Cleaned preview")
        st.dataframe(analysis.processed.head(80), use_container_width=True)
    with c2:
        st.write("Statistical report")
        st.dataframe(analysis.summary, use_container_width=True)

    numeric = analysis.numeric
    if not numeric.empty:
        corr = numeric.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Turbo",
            title="Lithography Feature Correlation Heatmap",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        pca_fig = px.scatter(
            analysis.pca,
            x="PC1",
            y="PC2",
            color=analysis.pca["anomaly_flag"].astype(str),
            title="PCA Process Manifold with Anomaly Labels",
            template="plotly_dark",
            labels={"color": "Anomaly"},
        )
        st.plotly_chart(pca_fig, use_container_width=True)

    st.download_button(
        "Download processed analytics CSV",
        dataframe_to_csv_bytes(analysis.processed),
        file_name="lithovision_processed_analytics.csv",
        mime="text/csv",
    )

with tabs[2]:
    st.subheader("Synthetic Wafer Data Generator")
    st.caption("Generate 10,000+ rows of synthetic lithography telemetry with drift, noise, and controlled defect classes.")
    generated = st.session_state.synthetic_df
    st.success(f"Synthetic dataset ready: {len(generated):,} rows saved to {st.session_state.synthetic_path}")
    st.dataframe(generated.head(100), use_container_width=True)

    defect_counts = generated["defect_class"].value_counts().reset_index()
    defect_counts.columns = ["defect_class", "count"]
    fig = px.bar(
        defect_counts,
        x="defect_class",
        y="count",
        color="defect_class",
        title="Synthetic Defect Class Simulator",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    heatmap_df = generated.pivot_table(
        values="defect_probability",
        index="wafer_id",
        columns=pd.cut(generated["process_step"], bins=24, labels=False),
        aggfunc="mean",
    )
    heatmap = go.Figure(data=go.Heatmap(z=heatmap_df.values, colorscale="Inferno"))
    heatmap.update_layout(title="Interactive Defect Heatmap by Wafer and Process Window", template="plotly_dark")
    st.plotly_chart(heatmap, use_container_width=True)

    st.download_button(
        "Download synthetic lithography dataset",
        dataframe_to_csv_bytes(generated),
        file_name="synthetic_lithography_dataset.csv",
        mime="text/csv",
    )

with tabs[3]:
    st.subheader("Lithography Image Reconstruction")
    st.caption("Upload a blurry, noisy, SEM-like, or wafer image for denoising, edge restoration, and defect highlighting.")
    if uploaded_image is None:
        st.warning("Upload an image in the sidebar to run reconstruction.")
    else:
        image = Image.open(uploaded_image)
        with st.spinner("Running OpenCV denoising, edge restoration, and defect localization..."):
            recon = reconstruct_lithography_image(image)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(recon.original, caption="Original grayscale", use_container_width=True, clamp=True)
        with c2:
            st.image(recon.reconstructed, caption="Reconstructed image", use_container_width=True, clamp=True)
        with c3:
            st.image(recon.highlighted, caption="Defect-highlighted reconstruction", use_container_width=True, clamp=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original sharpness", recon.metrics["original_sharpness"])
        m2.metric("Reconstructed sharpness", recon.metrics["reconstructed_sharpness"])
        m3.metric("Similarity score", f"{recon.metrics['similarity_score']:.2%}")
        m4.metric("AI confidence", f"{recon.metrics['ai_reconstruction_confidence']:.2%}")
        st.metric("Defect area percent", f"{recon.metrics['defect_area_percent']:.2f}%")
        buffer = BytesIO()
        Image.fromarray(recon.reconstructed).save(buffer, format="PNG")
        st.download_button("Download reconstructed image", buffer.getvalue(), "reconstructed_lithography.png", "image/png")

with tabs[4]:
    st.subheader("AI Insights & Exportable Report")
    st.caption(f"Mode: {inference_mode}")
    for insight in analysis.insights:
        st.markdown(f"- {insight}")
    st.divider()
    report = build_text_report(analysis.processed, analysis.insights, analysis.drift_status)
    st.text_area("Generated report", report, height=260)
    st.download_button("Export statistical report", report, "lithovision_ai_report.txt", "text/plain")
