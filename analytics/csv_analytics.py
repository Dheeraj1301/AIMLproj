"""CSV analytics and lightweight ML inference for lithography metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from synthetic.generator import LITHOGRAPHY_FEATURES


@dataclass
class AnalyticsResult:
    """Container for processed tabular analytics outputs."""

    processed: pd.DataFrame
    numeric: pd.DataFrame
    summary: pd.DataFrame
    pca: pd.DataFrame
    insights: list[str]
    yield_score: float
    defect_probability: float
    drift_status: str
    model_accuracy: float | None


def clean_lithography_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and fill missing values without dropping user data."""

    cleaned = df.copy()
    cleaned.columns = [str(col).strip().lower().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in cleaned.columns if col not in numeric_cols]

    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        cleaned[numeric_cols] = imputer.fit_transform(cleaned[numeric_cols])
    for col in categorical_cols:
        mode = cleaned[col].mode(dropna=True)
        cleaned[col] = cleaned[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    return cleaned


def _numeric_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    preferred = [feature for feature in LITHOGRAPHY_FEATURES if feature in numeric.columns]
    if preferred:
        return numeric[preferred]
    return numeric


def detect_anomalies(numeric: pd.DataFrame) -> np.ndarray:
    """Run Isolation Forest and return 1 for anomalies and 0 for nominal rows."""

    if numeric.empty or len(numeric) < 5:
        return np.zeros(len(numeric), dtype=int)
    model = IsolationForest(n_estimators=160, contamination="auto", random_state=42)
    labels = model.fit_predict(StandardScaler().fit_transform(numeric))
    return (labels == -1).astype(int)


def pca_projection(numeric: pd.DataFrame) -> pd.DataFrame:
    """Project numeric features into two PCA components for visualization."""

    if numeric.empty:
        return pd.DataFrame(columns=["PC1", "PC2"])
    if numeric.shape[1] == 1:
        return pd.DataFrame({"PC1": numeric.iloc[:, 0], "PC2": np.zeros(len(numeric))})
    scaled = StandardScaler().fit_transform(numeric)
    components = PCA(n_components=2, random_state=42).fit_transform(scaled)
    return pd.DataFrame(components, columns=["PC1", "PC2"])


def predict_defects(processed: pd.DataFrame, numeric: pd.DataFrame) -> tuple[np.ndarray, float | None]:
    """Train a lightweight RandomForest when labels exist, otherwise infer risk heuristically."""

    if "defect_label" in processed.columns and processed["defect_label"].nunique() > 1 and len(processed) >= 20:
        y = processed["defect_label"].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            numeric, y, test_size=0.25, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
        model.fit(x_train, y_train)
        proba = model.predict_proba(numeric)[:, 1]
        accuracy = accuracy_score(y_test, model.predict(x_test))
        return proba, float(accuracy)

    if numeric.empty:
        return np.zeros(len(processed)), None
    zscores = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1))
    risk = np.clip(zscores.mean(axis=1) / 3.0, 0, 1)
    return risk.to_numpy(), None


def calculate_yield_score(defect_probability: pd.Series, anomaly_flag: pd.Series) -> pd.Series:
    """Calculate an interpretable wafer quality/yield score."""

    return np.clip(100 - defect_probability * 62 - anomaly_flag * 14, 0, 100).round(2)


def process_drift_analysis(numeric: pd.DataFrame) -> tuple[str, float]:
    """Compare early and late process windows to estimate process drift."""

    if len(numeric) < 20 or numeric.empty:
        return "Insufficient data for process drift analysis", 0.0
    window = max(10, len(numeric) // 5)
    early = numeric.head(window).mean()
    late = numeric.tail(window).mean()
    scale = numeric.std(ddof=0).replace(0, 1)
    drift_score = float(np.abs((late - early) / scale).mean())
    if drift_score > 0.85:
        status = "High process drift detected — review exposure dose and overlay alignment."
    elif drift_score > 0.45:
        status = "Moderate process drift detected — monitor lot-to-lot stability."
    else:
        status = "Process drift is stable within synthetic control limits."
    return status, drift_score


def generate_insights(processed: pd.DataFrame, numeric: pd.DataFrame, drift_score: float) -> list[str]:
    """Create concise, dashboard-ready semiconductor analytics insights."""

    insights: list[str] = []
    if "defect_probability" in processed:
        high_risk = float((processed["defect_probability"] > 0.6).mean() * 100)
        insights.append(f"{high_risk:.1f}% of wafers are above the high-risk defect probability threshold.")
    if "yield_score" in processed:
        insights.append(f"Average yield optimization score is {processed['yield_score'].mean():.1f}/100.")
    if not numeric.empty:
        strongest_variance = numeric.std().sort_values(ascending=False).index[0]
        insights.append(f"{strongest_variance} shows the strongest process variance contribution.")
    insights.append(f"Process drift index is {drift_score:.2f}, supporting process-aware anomaly inference.")
    if "anomaly_flag" in processed:
        insights.append(f"Isolation Forest marked {int(processed['anomaly_flag'].sum())} anomalous process records.")
    return insights


def analyze_csv(df: pd.DataFrame) -> AnalyticsResult:
    """Run the complete CSV analytics pipeline."""

    processed = clean_lithography_csv(df)
    numeric = _numeric_feature_frame(processed)
    processed["anomaly_flag"] = detect_anomalies(numeric) if not numeric.empty else np.zeros(len(processed), dtype=int)
    defect_probability, accuracy = predict_defects(processed, numeric)
    processed["defect_probability"] = np.round(defect_probability, 4)
    processed["wafer_quality_score"] = calculate_yield_score(
        processed["defect_probability"], processed["anomaly_flag"]
    )
    drift_status, drift_score = process_drift_analysis(numeric)
    processed["process_drift_score"] = round(drift_score, 4)
    pca = pca_projection(numeric)
    pca["anomaly_flag"] = processed["anomaly_flag"].to_numpy() if len(processed) else []
    insights = generate_insights(processed, numeric, drift_score)
    return AnalyticsResult(
        processed=processed,
        numeric=numeric,
        summary=processed.describe(include="all").transpose(),
        pca=pca,
        insights=insights,
        yield_score=float(processed["wafer_quality_score"].mean()) if len(processed) else 0.0,
        defect_probability=float(processed["defect_probability"].mean()) if len(processed) else 0.0,
        drift_status=drift_status,
        model_accuracy=accuracy,
    )
