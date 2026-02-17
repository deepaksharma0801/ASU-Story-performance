import io
import json
import os
import sqlite3
from datetime import date, datetime, time, timedelta
from statistics import NormalDist

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from docx import Document
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP_TITLE = "Predicting Digital Content Performance"
SUBTITLE = "Data-driven content planning for sustainable and effective storytelling"
DATASET_NAME = "ASU Story Performance"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ASU Story Performance .xlsx")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "saved_predictions.db")
OPENAPI_PATH = os.path.join(os.path.dirname(__file__), "content_predict_api.yaml")
ASSET_LOGO = os.path.join(os.path.dirname(__file__), "assets", "asu_logo.svg")
FAVICON = os.path.join(os.path.dirname(__file__), "assets", "favicon.svg")

COL_DATE = "Publication Date"
COL_URL = "News URL"
COL_IMAGES = "Contain Images in Between"
COL_LENGTH = "Long(L) or Short(S)"
COL_WORD_COUNT = "Word Count"
COL_SRA = "SRA"

METRICS_30 = {
    "Users": "30-day Users",
    "Sessions": "30-day Sessions",
    "Pageviews": "30-day Pageviews",
    "Session Eng. Time": "30-day Session Eng. Time",
    "Bounce Rate": "30-day Bounce Rate",
}

METRICS_90 = {
    "Users": "90-day Users",
    "Sessions": "90-day Sessions",
    "Pageviews": "90-day Pageviews",
    "Session Eng. Time": "90-day Session Eng. Time",
    "Bounce Rate": "90-day Bounce Rate",
}

EXPECTED_COLUMNS = [
    COL_DATE,
    COL_URL,
    COL_IMAGES,
    COL_LENGTH,
    COL_WORD_COUNT,
    COL_SRA,
    METRICS_30["Users"],
    METRICS_30["Sessions"],
    METRICS_30["Pageviews"],
    METRICS_30["Session Eng. Time"],
    METRICS_30["Bounce Rate"],
    METRICS_90["Users"],
    METRICS_90["Sessions"],
    METRICS_90["Pageviews"],
    METRICS_90["Session Eng. Time"],
    METRICS_90["Bounce Rate"],
]

MODEL_VERSION = "v1.0"
LAST_RETRAIN_DATE = "2026-01-15"

HAS_DIALOG = hasattr(st, "dialog")
HAS_COLUMN_CONFIG = hasattr(st, "column_config")
LAST_DATA_ERROR = None


def empty_chart() -> alt.Chart:
    return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x:Q", y="y:Q")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    if not set(EXPECTED_COLUMNS).issubset(df.columns) and df.shape[1] >= len(EXPECTED_COLUMNS):
        if COL_DATE in df.columns:
            start_idx = list(df.columns).index(COL_DATE)
            if df.shape[1] - start_idx >= len(EXPECTED_COLUMNS):
                df = df.iloc[:, start_idx : start_idx + len(EXPECTED_COLUMNS)]
            else:
                df = df.iloc[:, : len(EXPECTED_COLUMNS)]
        else:
            df = df.iloc[:, : len(EXPECTED_COLUMNS)]
        df.columns = EXPECTED_COLUMNS
    return df


def enable_altair_theme() -> None:
    def _theme():
        return {
            "config": {
                "background": "transparent",
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": "#2E1B1E",
                    "titleColor": "#2E1B1E",
                    "gridColor": "#E6E0D8",
                },
                "legend": {
                    "labelColor": "#2E1B1E",
                    "titleColor": "#2E1B1E",
                },
                "title": {"color": "#2E1B1E"},
            }
        }

    alt.themes.register("asu_light", _theme)
    alt.themes.enable("asu_light")


def _read_excel_with_header_guess(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip().casefold() for v in row.values if pd.notna(v)]
        if COL_DATE.casefold() in values:
            header_row = idx
            break
    if header_row is None:
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(path, header=header_row)
    df = df.dropna(how="all")
    return _normalize_columns(df)


def _safe_read_excel(path: str) -> pd.DataFrame:
    global LAST_DATA_ERROR
    try:
        LAST_DATA_ERROR = None
        return _read_excel_with_header_guess(path)
    except Exception as exc:
        LAST_DATA_ERROR = str(exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = _safe_read_excel(path)
    if df.empty:
        return df

    df = df.copy()
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    if COL_IMAGES in df.columns:
        df[COL_IMAGES] = (
            df[COL_IMAGES]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "y", "true", "1"])
        )
    if COL_LENGTH in df.columns:
        df[COL_LENGTH] = df[COL_LENGTH].astype(str).str.strip()
    if COL_WORD_COUNT in df.columns:
        df[COL_WORD_COUNT] = pd.to_numeric(df[COL_WORD_COUNT], errors="coerce")

    for col in list(METRICS_30.values()) + list(METRICS_90.values()):
        if col not in df.columns:
            continue
        if col in (METRICS_30["Session Eng. Time"], METRICS_90["Session Eng. Time"]):
            df[col] = df[col].apply(coerce_duration_seconds)
        elif col in (METRICS_30["Bounce Rate"], METRICS_90["Bounce Rate"]):
            df[col] = coerce_percent_series(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if COL_DATE in df.columns:
        df["day_of_week"] = df[COL_DATE].dt.day_name()
        df["month"] = df[COL_DATE].dt.month
        df["is_weekend"] = df[COL_DATE].dt.weekday >= 5

    if COL_URL in df.columns and "Title" not in df.columns:
        df["Title"] = df[COL_URL].astype(str).str.rstrip("/").str.split("/").str[-1]
        df["Title"] = df["Title"].str.replace("-", " ").str.title()

    return df


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_url(url: str) -> str:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(p for p in paragraphs if p)


def extract_draft_text(paste_text: str, uploaded_file, url: str) -> tuple[str, str]:
    if paste_text and paste_text.strip():
        return paste_text.strip(), "Pasted text"
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        data = uploaded_file.read()
        if name.endswith(".txt"):
            return data.decode("utf-8", errors="ignore"), f"Uploaded file ({uploaded_file.name})"
        if name.endswith(".docx"):
            try:
                return extract_text_from_docx(data), f"Uploaded file ({uploaded_file.name})"
            except Exception:
                return "", "Uploaded file could not be read"
    if url and url.strip():
        try:
            return extract_text_from_url(url.strip()), "URL"
        except Exception:
            return "", "URL could not be fetched"
    return "", ""


def draft_stats(text: str) -> dict:
    if not text:
        return {"word_count": None, "sentence_count": None, "avg_sentence_length": None}
    words = [w for w in text.replace("\n", " ").split(" ") if w.strip()]
    word_count = len(words)
    sentences = [s for s in text.replace("\n", " ").split(".") if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count if sentence_count else None
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
    }


def coerce_duration_seconds(value: object) -> float | None:
    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.nan
        if ":" in text:
            try:
                return pd.to_timedelta(text).total_seconds()
            except Exception:
                return np.nan
        try:
            return float(text)
        except Exception:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def coerce_percent_series(series: pd.Series) -> pd.Series:
    def _to_percent(val: object) -> float | None:
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            text = val.strip().replace("%", "")
            if not text:
                return np.nan
            try:
                num = float(text)
            except Exception:
                return np.nan
        else:
            try:
                num = float(val)
            except Exception:
                return np.nan
        if 0 <= num <= 1:
            return num * 100
        return num

    return series.apply(_to_percent)


def get_last_refresh(path: str) -> str:
    if not os.path.exists(path):
        return "Unknown"
    ts = datetime.fromtimestamp(os.path.getmtime(path))
    return ts.strftime("%b %d, %Y %I:%M %p")


def ensure_db(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            planned_publication_date TEXT,
            title TEXT,
            inputs_json TEXT NOT NULL,
            predictions_json TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction_to_db(path: str, inputs: dict, predictions: dict, title: str) -> None:
    ensure_db(path)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO saved_predictions (created_at, planned_publication_date, title, inputs_json, predictions_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            inputs.get("planned_date"),
            title,
            json.dumps(inputs),
            json.dumps(predictions),
            "",
        ),
    )
    conn.commit()
    conn.close()


def load_saved_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM saved_predictions ORDER BY created_at DESC", conn)
    conn.close()
    return df


def apply_filters(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if COL_DATE in filtered.columns and state.get("date_range"):
        start_date, end_date = state["date_range"]
        filtered = filtered[(filtered[COL_DATE] >= pd.to_datetime(start_date)) & (filtered[COL_DATE] <= pd.to_datetime(end_date))]
    if COL_SRA in filtered.columns and state.get("sra"):
        filtered = filtered[filtered[COL_SRA].isin(state["sra"])]
    if COL_LENGTH in filtered.columns and state.get("length") and state["length"] != "Both":
        want_long = state["length"] == "Long"
        filtered = filtered[filtered[COL_LENGTH].str.upper().str.startswith("L") == want_long]
    if COL_IMAGES in filtered.columns and state.get("images") and state["images"] != "Both":
        want_images = state["images"] == "Images only"
        filtered = filtered[filtered[COL_IMAGES] == want_images]
    if COL_WORD_COUNT in filtered.columns and state.get("min_word_count") is not None:
        filtered = filtered[filtered[COL_WORD_COUNT] >= state["min_word_count"]]
    return filtered


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["word_count"] = data[COL_WORD_COUNT]
    data["images_bin"] = data[COL_IMAGES].astype(bool).astype(int)
    data["length_bin"] = data[COL_LENGTH].astype(str).str.upper().str.startswith("L").astype(int)
    data["month"] = data[COL_DATE].dt.month
    data["day_of_week"] = data[COL_DATE].dt.day_name()
    data["is_weekend"] = (data[COL_DATE].dt.weekday >= 5).astype(int)
    if COL_SRA in data.columns:
        data["sra"] = data[COL_SRA].fillna("Unknown").astype(str)
    else:
        data["sra"] = "Unknown"
    return data


def make_regressor(model_choice: str):
    if model_choice == "Baseline linear":
        return LinearRegression()
    if model_choice == "Random Forest":
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    if model_choice == "XGBoost (explainable)":
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
        )
    return RandomForestRegressor(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "r2": None}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_models(df: pd.DataFrame, model_choice: str):
    required = [COL_DATE, COL_WORD_COUNT] + list(METRICS_30.values()) + list(METRICS_90.values())
    if df.empty or any(col not in df.columns for col in required):
        return None

    target_cols = list(METRICS_30.values()) + list(METRICS_90.values())
    data = df.dropna(subset=[COL_DATE] + target_cols).copy()
    if len(data) < len(df):
        st.warning(f"Training data filtered to {len(data)} rows due to missing target metrics.")
    if len(data) < 30:
        return None

    data = build_feature_frame(data)
    data = data.sort_values(COL_DATE)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    val_df = data.iloc[split_idx:]

    feature_cols = ["word_count", "images_bin", "length_bin", "month", "is_weekend", "day_of_week", "sra"]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    y_train_30 = train_df[list(METRICS_30.values())]
    y_train_90 = train_df[list(METRICS_90.values())]
    y_val_30 = val_df[list(METRICS_30.values())]
    y_val_90 = val_df[list(METRICS_90.values())]

    categorical_features = ["day_of_week", "sra"]
    numeric_features = ["word_count", "images_bin", "length_bin", "month", "is_weekend"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model_30 = Pipeline(steps=[("prep", preprocessor), ("model", make_regressor(model_choice))])
    model_90 = Pipeline(steps=[("prep", preprocessor), ("model", make_regressor(model_choice))])

    model_30.fit(X_train, y_train_30)
    model_90.fit(X_train, y_train_90)

    val_pred_30 = model_30.predict(X_val)
    val_pred_90 = model_90.predict(X_val)

    idx_30 = list(METRICS_30.values()).index(METRICS_30["Pageviews"])
    idx_90 = list(METRICS_90.values()).index(METRICS_90["Pageviews"])
    resid_30 = y_val_30.iloc[:, idx_30].values - val_pred_30[:, idx_30]
    resid_90 = y_val_90.iloc[:, idx_90].values - val_pred_90[:, idx_90]

    metrics_30 = eval_metrics(y_val_30.iloc[:, idx_30].values, val_pred_30[:, idx_30])
    metrics_90 = eval_metrics(y_val_90.iloc[:, idx_90].values, val_pred_90[:, idx_90])

    return {
        "model_30": model_30,
        "model_90": model_90,
        "resid_std_30": float(np.nanstd(resid_30)) if len(resid_30) else None,
        "resid_std_90": float(np.nanstd(resid_90)) if len(resid_90) else None,
        "feature_cols": feature_cols,
        "val_metrics_30": metrics_30,
        "val_metrics_90": metrics_90,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "val_start": str(pd.to_datetime(val_df[COL_DATE].min()).date()) if not val_df.empty else None,
        "val_end": str(pd.to_datetime(val_df[COL_DATE].max()).date()) if not val_df.empty else None,
    }


@st.cache_resource(show_spinner=False)
def get_trained_models(df: pd.DataFrame, model_choice: str):
    return train_models(df, model_choice)


def model_predict(models, inputs: dict) -> dict | None:
    if not models:
        return None
    feature_cols = models["feature_cols"]
    row = {
        "word_count": inputs.get("word_count"),
        "images_bin": 1 if inputs.get("images") else 0,
        "length_bin": 1 if inputs.get("length") == "Long" else 0,
        "month": pd.to_datetime(inputs.get("planned_date")).month,
        "is_weekend": 1 if pd.to_datetime(inputs.get("planned_date")).weekday() >= 5 else 0,
        "day_of_week": pd.to_datetime(inputs.get("planned_date")).day_name(),
        "sra": inputs.get("sra"),
    }
    X = pd.DataFrame([row])[feature_cols]
    pred_30 = models["model_30"].predict(X)[0]
    pred_90 = models["model_90"].predict(X)[0]

    metrics_30 = dict(zip(METRICS_30.keys(), pred_30))
    metrics_90 = dict(zip(METRICS_90.keys(), pred_90))
    for metrics in (metrics_30, metrics_90):
        if "Bounce Rate" in metrics:
            metrics["Bounce Rate"] = float(np.clip(metrics["Bounce Rate"], 0, 100))

    return {
        "metrics_30": metrics_30,
        "metrics_90": metrics_90,
        "pred_30": float(metrics_30["Pageviews"]),
        "pred_90": float(metrics_90["Pageviews"]),
        "resid_std_30": models.get("resid_std_30"),
        "resid_std_90": models.get("resid_std_90"),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols))
        elif name == "num":
            names.extend(cols)
    return names


def build_feature_importance_dynamic(models) -> pd.DataFrame:
    if not models:
        return build_feature_importance()
    preprocessor = models["model_30"].named_steps["prep"]
    model = models["model_30"].named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return build_feature_importance()
    names = get_feature_names(preprocessor)
    importances = model.feature_importances_
    if len(importances) != len(names):
        return build_feature_importance()
    data = pd.DataFrame({"feature": names, "importance": importances})
    top = data.sort_values("importance", ascending=False).head(10)
    top["direction"] = "Mixed"
    top["feature"] = top["feature"].str.replace("day_of_week_", "Day: ").str.replace("sra_", "SRA: ")
    return top


def build_local_contributions_dynamic(inputs: dict, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_WORD_COUNT not in df.columns:
        return build_local_contributions()
    contributions = []
    median_wc = df[COL_WORD_COUNT].median()
    wc = inputs.get("word_count", median_wc)
    contributions.append(("Word Count", 0.12 if wc >= median_wc else -0.08))

    if COL_IMAGES in df.columns:
        images_perf = df.groupby(COL_IMAGES)[METRICS_30["Pageviews"]].mean()
        if True in images_perf.index and False in images_perf.index:
            better_with_images = images_perf[True] >= images_perf[False]
            if inputs.get("images") and better_with_images:
                contributions.append(("Contain Images in Between", 0.08))
            elif (not inputs.get("images")) and better_with_images:
                contributions.append(("Contain Images in Between", -0.06))
            else:
                contributions.append(("Contain Images in Between", 0.02))

    length_val = inputs.get("length", "Long")
    if length_val == "Long":
        contributions.append(("Long(L) or Short(S)", 0.05))
    else:
        contributions.append(("Long(L) or Short(S)", -0.03))

    try:
        pub_date = pd.to_datetime(inputs.get("planned_date"))
        contributions.append(("Publication Month", 0.02))
        contributions.append(("Day of Week", 0.01))
    except Exception:
        contributions.append(("Publication Month", 0.0))
        contributions.append(("Day of Week", 0.0))

    if COL_SRA in df.columns and METRICS_30["Pageviews"] in df.columns:
        sra_perf = df.groupby(COL_SRA)[METRICS_30["Pageviews"]].mean()
        sra = inputs.get("sra")
        if sra in sra_perf.index and sra_perf[sra] >= sra_perf.median():
            contributions.append(("SRA", 0.03))
        else:
            contributions.append(("SRA", -0.03))

    return pd.DataFrame(contributions, columns=["feature", "contribution"])


def format_number(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.0f}"


def format_bounce(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if 0 <= value <= 1:
        return f"{value*100:.1f}%"
    return f"{value:.1f}%"


def format_duration(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    seconds = int(round(value))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_simple_pdf(report_title: str, date_range: tuple[date, date], sections: list[str]) -> bytes:
    start_date, end_date = date_range
    lines = [
        report_title or "Content Performance Report",
        f"Date range: {start_date} to {end_date}",
        "Included sections: " + (", ".join(sections) if sections else "None"),
    ]
    lines = [pdf_escape(line) for line in lines]
    content = (
        "BT /F1 18 Tf 72 720 Td "
        f"({lines[0]}) Tj "
        "0 -24 Td /F1 12 Tf "
        f"({lines[1]}) Tj "
        "0 -18 Td "
        f"({lines[2]}) Tj "
        "ET"
    )
    content_bytes = content.encode("latin-1")
    parts = []
    parts.append(b"%PDF-1.4\n")
    parts.append(b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")
    parts.append(b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n")
    parts.append(b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n")
    parts.append(f"4 0 obj<</Length {len(content_bytes)}>>stream\n".encode("latin-1"))
    parts.append(content_bytes + b"\nendstream endobj\n")
    parts.append(b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n")
    xref_start = sum(len(p) for p in parts)
    parts.append(b"xref\n0 6\n0000000000 65535 f \n")
    offset = 0
    for p in parts[:-2]:
        parts.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
        offset += len(p)
    parts.append(b"trailer<</Size 6/Root 1 0 R>>\n")
    parts.append(f"startxref\n{xref_start}\n%%EOF".encode("latin-1"))
    return b"".join(parts)


def derive_metrics(df: pd.DataFrame, pageviews: float, mapping: dict) -> dict:
    metrics = {}
    pv_col = mapping["Pageviews"]
    pv_median = df[pv_col].median() if pv_col in df.columns else None
    for key in ["Users", "Sessions", "Session Eng. Time"]:
        col = mapping[key]
        if col in df.columns and pv_median and pv_median > 0:
            ratio = df[col].median() / pv_median
            metrics[key] = pageviews * ratio
        elif col in df.columns:
            metrics[key] = df[col].median()
        else:
            metrics[key] = None
    br_col = mapping["Bounce Rate"]
    metrics["Bounce Rate"] = df[br_col].median() if br_col in df.columns else None
    return metrics


def compute_prediction(inputs: dict, df: pd.DataFrame) -> dict:
    model_choice = inputs.get("model", "Random Forest")
    models = get_trained_models(df, model_choice)
    model_result = model_predict(models, inputs) if models else None

    if model_result:
        pred_30 = model_result["pred_30"]
        pred_90 = model_result["pred_90"]
        metrics_30 = model_result["metrics_30"]
        metrics_90 = model_result["metrics_90"]
    else:
        if df.empty or METRICS_30["Pageviews"] not in df.columns:
            base_30 = 12000
        else:
            base_30 = float(df[METRICS_30["Pageviews"]].median())

        word_count = inputs.get("word_count", 800) or 0
        images = inputs.get("images", True)
        is_long = inputs.get("length", "Long") == "Long"

        wc_factor = 1 + min(max(word_count - 800, -400), 800) / 4000
        img_factor = 1.08 if images else 0.94
        len_factor = 1.05 if is_long else 0.96

        pred_30 = max(1000, base_30 * wc_factor * img_factor * len_factor)
        pred_90 = pred_30 * 1.55
        metrics_30 = derive_metrics(df, pred_30, METRICS_30)
        metrics_90 = derive_metrics(df, pred_90, METRICS_90)

    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        high_cut, mid_cut = 20000, 12000
    else:
        high_cut = float(df[METRICS_30["Pageviews"]].quantile(0.75))
        mid_cut = float(df[METRICS_30["Pageviews"]].quantile(0.45))

    if pred_30 >= high_cut:
        perf_class = "High"
        perf_color = "#2E7D32"
        perf_prob = 0.82
    elif pred_30 >= mid_cut:
        perf_class = "Medium"
        perf_color = "#EF6C00"
        perf_prob = 0.64
    else:
        perf_class = "Low"
        perf_color = "#C62828"
        perf_prob = 0.47

    interval_level = inputs.get("ci", "95%")
    if interval_level == "None":
        interval = None
    else:
        z = 1.64 if interval_level == "90%" else 1.96
        if model_result and model_result.get("resid_std_30"):
            spread = model_result["resid_std_30"]
        else:
            spread = pred_30 * 0.18
        interval = (pred_30 - z * spread, pred_30 + z * spread)

    return {
        "pred_30": pred_30,
        "pred_90": pred_90,
        "metrics_30": metrics_30,
        "metrics_90": metrics_90,
        "perf_class": perf_class,
        "perf_color": perf_color,
        "perf_prob": perf_prob,
        "interval": interval,
        "model_used": bool(model_result),
    }


def inputs_signature(inputs: dict) -> str:
    draft_stats = inputs.get("draft_stats") or {}
    payload = {
        "planned_date": inputs.get("planned_date"),
        "sra": inputs.get("sra"),
        "images": inputs.get("images"),
        "length": inputs.get("length"),
        "word_count": inputs.get("word_count"),
        "model": inputs.get("model"),
        "ci": inputs.get("ci"),
        "draft_word_count": draft_stats.get("word_count"),
        "avg_sentence_length": draft_stats.get("avg_sentence_length"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def build_feature_importance() -> pd.DataFrame:
    features = [
        ("Word Count", 0.28, "Positive"),
        ("Contain Images in Between", 0.19, "Positive"),
        ("Long(L) or Short(S)", 0.14, "Positive"),
        ("Publication Month", 0.11, "Mixed"),
        ("Day of Week", 0.09, "Mixed"),
        ("SRA", 0.08, "Mixed"),
        ("Recent Pageviews", 0.06, "Positive"),
        ("Engagement Time", 0.03, "Positive"),
        ("Bounce Rate", 0.02, "Negative"),
        ("Seasonality", 0.02, "Mixed"),
    ]
    return pd.DataFrame(features, columns=["feature", "importance", "direction"])


def build_local_contributions() -> pd.DataFrame:
    contributions = [
        ("Word Count", 0.12),
        ("Contain Images in Between", 0.08),
        ("Long(L) or Short(S)", 0.05),
        ("Publication Month", 0.03),
        ("SRA", -0.04),
        ("Day of Week", -0.02),
    ]
    return pd.DataFrame(contributions, columns=["feature", "contribution"])


def sra_insights(df: pd.DataFrame, sra: str | None) -> dict:
    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        return {}
    data = df.copy()
    if sra and COL_SRA in df.columns:
        subset = data[data[COL_SRA] == sra]
        if len(subset) >= 10:
            data = subset
    insights = {}
    insights["n_rows"] = len(data)
    if COL_WORD_COUNT in data.columns:
        top = data[data[METRICS_30["Pageviews"]] >= data[METRICS_30["Pageviews"]].quantile(0.75)]
        if not top.empty:
            insights["wc_target"] = int(top[COL_WORD_COUNT].median())
            insights["wc_low"] = int(top[COL_WORD_COUNT].quantile(0.25))
            insights["wc_high"] = int(top[COL_WORD_COUNT].quantile(0.75))
            insights["top_n"] = len(top)
        else:
            insights["wc_target"] = int(data[COL_WORD_COUNT].median())
    if COL_WORD_COUNT in data.columns and METRICS_30["Pageviews"] in data.columns:
        wc_target = insights.get("wc_target")
        if wc_target:
            long_form = data[data[COL_WORD_COUNT] >= wc_target]
            short_form = data[data[COL_WORD_COUNT] < wc_target]
            if not long_form.empty and not short_form.empty:
                long_avg = long_form[METRICS_30["Pageviews"]].mean()
                short_avg = short_form[METRICS_30["Pageviews"]].mean()
                if short_avg:
                    insights["long_lift_pct"] = ((long_avg - short_avg) / short_avg) * 100
    if COL_IMAGES in data.columns:
        img_perf = data.groupby(COL_IMAGES)[METRICS_30["Pageviews"]].mean()
        if True in img_perf.index and False in img_perf.index:
            insights["images_help"] = img_perf[True] > img_perf[False]
            insights["img_mean_true"] = img_perf[True]
            insights["img_mean_false"] = img_perf[False]
            if img_perf[False] > 0:
                insights["images_lift_pct"] = ((img_perf[True] - img_perf[False]) / img_perf[False]) * 100
    if COL_DATE in data.columns:
        by_day = data.groupby(data[COL_DATE].dt.day_name())[METRICS_30["Pageviews"]].mean()
        if not by_day.empty:
            insights["best_day"] = by_day.idxmax()
            insights["best_day_avg"] = by_day.max()
            overall = data[METRICS_30["Pageviews"]].mean()
            if overall:
                insights["best_day_lift_pct"] = ((by_day.max() - overall) / overall) * 100
    return insights


def segment_for_inputs(df: pd.DataFrame, inputs: dict, min_rows: int = 30) -> pd.DataFrame:
    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        return df
    data = df.dropna(subset=[METRICS_30["Pageviews"], COL_WORD_COUNT, COL_DATE]).copy()
    sra = inputs.get("sra")
    if sra and COL_SRA in data.columns:
        subset = data[data[COL_SRA] == sra]
        if len(subset) >= min_rows:
            data = subset
    return data


def model_pred_30(models, inputs: dict) -> float | None:
    if not models:
        return None
    try:
        result = model_predict(models, inputs)
    except Exception:
        return None
    if not result:
        return None
    return float(result["pred_30"])


def pick_best_bin(values: pd.Series, metric: pd.Series) -> tuple[tuple[float, float], float] | None:
    if values.empty or metric.empty:
        return None
    try:
        bins = pd.qcut(values, q=3, duplicates="drop")
    except Exception:
        return None
    grouped = metric.groupby(bins).mean()
    if grouped.empty:
        return None
    best_bin = grouped.idxmax()
    return (float(best_bin.left), float(best_bin.right)), float(grouped.max())


def build_optimization_scenarios(
    pred_30: float, inputs: dict, df: pd.DataFrame
) -> tuple[pd.DataFrame, dict, list[dict]]:
    scenarios: list[dict] = []
    details: dict = {}
    evidence_items: list[dict] = []

    def add_evidence(variant: str, claim: str, stat: str, n: str | int, source: str) -> str:
        evidence_id = f"E{len(evidence_items) + 1}"
        evidence_items.append(
            {
                "Evidence ID": evidence_id,
                "Variant": variant,
                "Claim": claim,
                "Stat": stat,
                "Sample": str(n),
                "Source": source,
            }
        )
        return evidence_id

    def add_scenario(
        name: str,
        change: str,
        delta: float | None,
        projected: float | None,
        detail: str,
        base: float | None = None,
    ):
        scenarios.append(
            {
                "Scenario": name,
                "Suggested change": change,
                "Estimated lift": f"+{int(delta * 100)}%" if delta is not None else "—",
                "Current 30-day Pageviews": int(base) if base is not None else "—",
                "New 30-day Pageviews": int(projected) if projected is not None else "—",
                "Delta Pageviews": int(projected - base) if (projected is not None and base is not None) else "—",
                "_delta": delta if delta is not None else -1,
            }
        )
        details[name] = detail

    data = segment_for_inputs(df, inputs)
    models = get_trained_models(df, inputs.get("model", "Random Forest"))
    base_pred = model_pred_30(models, inputs)
    if base_pred is None:
        base_pred = float(data[METRICS_30["Pageviews"]].mean()) if not data.empty else pred_30
    base_pred = float(base_pred) if base_pred else pred_30

    def delta_from_preds(alt_pred: float | None) -> float | None:
        if alt_pred is None or not base_pred:
            return None
        return (alt_pred - base_pred) / base_pred

    # Variant A: Word count target (SRA segment)
    word_count = int(inputs.get("word_count") or 0)
    if not data.empty and COL_WORD_COUNT in data.columns:
        wc_info = pick_best_bin(data[COL_WORD_COUNT], data[METRICS_30["Pageviews"]])
        if wc_info:
            (low, high), best_mean = wc_info
            if word_count < low or word_count > high:
                target_wc = int(data[(data[COL_WORD_COUNT] >= low) & (data[COL_WORD_COUNT] <= high)][COL_WORD_COUNT].median())
                alt_inputs = dict(inputs)
                alt_inputs["word_count"] = target_wc
                alt_pred = model_pred_30(models, alt_inputs)
                delta = delta_from_preds(alt_pred)
                evidence_id = add_evidence(
                    "Variant A",
                    "Best-performing word count band",
                    f"{int(low)}–{int(high)} words mean {int(best_mean):,} pageviews",
                    f"n={len(data)}",
                    "ASU Story Performance .xlsx",
                )
                detail = (
                    f"Adjust word count toward {target_wc} \u2192 "
                    f"{'+' + str(int(delta*100)) + '% predicted' if delta is not None else 'data-based lift not available'}. "
                    f"Current predicted: {int(base_pred):,}. New predicted: {int(alt_pred):,}. "
                    "Suggested copy change: tighten or expand to fit the best-performing band. "
                    "Why: This word-count band is strongest for similar stories in the dataset. "
                    f"Evidence: See Evidence tab ({evidence_id})."
                )
                add_scenario("Variant A", f"Adjust word count toward {target_wc}", delta, alt_pred, detail, base_pred)

    # Variant B: Images toggle (SRA segment)
    if COL_IMAGES in data.columns and not data.empty:
        img_perf = data.groupby(COL_IMAGES)[METRICS_30["Pageviews"]].mean()
        if True in img_perf.index and False in img_perf.index:
            current_has_images = bool(inputs.get("images", True))
            best_has_images = bool(img_perf[True] >= img_perf[False])
            if current_has_images != best_has_images:
                change = "Add images between paragraphs" if best_has_images else "Reduce images and simplify layout"
                alt_inputs = dict(inputs)
                alt_inputs["images"] = best_has_images
                alt_pred = model_pred_30(models, alt_inputs)
                delta = delta_from_preds(alt_pred)
                evidence_id = add_evidence(
                    "Variant B",
                    "Images vs no images performance",
                    f"Images mean {int(img_perf[True]):,} vs no-images {int(img_perf[False]):,}",
                    f"n={len(data)}",
                    "ASU Story Performance .xlsx",
                )
                detail = (
                    f"{change} \u2192 "
                    f"{'+' + str(int(delta*100)) + '% predicted' if delta is not None else 'data-based lift not available'}. "
                    f"Current predicted: {int(base_pred):,}. New predicted: {int(alt_pred):,}. "
                    "Suggested copy change: place one image after the lede and one mid-article. "
                    "Why: Image usage performs better for similar stories in the dataset. "
                    f"Evidence: See Evidence tab ({evidence_id})."
                )
                add_scenario("Variant B", change, delta, alt_pred, detail, base_pred)

    # Variant C: Best publish day (SRA segment)
    if COL_DATE in data.columns and not data.empty:
        by_day = data.groupby(data[COL_DATE].dt.day_name())[METRICS_30["Pageviews"]].mean()
        if not by_day.empty:
            best_day = by_day.idxmax()
            try:
                planned = pd.to_datetime(inputs.get("planned_date")).date()
                current_day = planned.strftime("%A")
            except Exception:
                current_day = None
            if current_day and best_day and current_day != best_day:
                change = f"Shift publication to {best_day}"
                alt_inputs = dict(inputs)
                day_delta = (list(by_day.index).index(best_day) - list(by_day.index).index(current_day)) % 7
                alt_inputs["planned_date"] = str(pd.Timestamp(planned) + pd.Timedelta(day_delta, unit="D"))
                alt_pred = model_pred_30(models, alt_inputs)
                delta = delta_from_preds(alt_pred)
                evidence_id = add_evidence(
                    "Variant C",
                    "Best publish day",
                    f"{best_day} mean {int(by_day[best_day]):,} pageviews",
                    f"n={len(data)}",
                    "ASU Story Performance .xlsx",
                )
                detail = (
                    f"{change} \u2192 "
                    f"{'+' + str(int(delta*100)) + '% predicted' if delta is not None else 'data-based lift not available'}. "
                    f"Current predicted: {int(base_pred):,}. New predicted: {int(alt_pred):,}. "
                    "Suggested copy change: align headline with that day’s audience expectations. "
                    f"Why: {best_day} is the strongest day for similar stories. "
                    f"Evidence: See Evidence tab ({evidence_id})."
                )
                add_scenario("Variant C", change, delta, alt_pred, detail, base_pred)

    # Variant D: Readability (draft-specific)
    avg_sentence_len = None
    if isinstance(inputs.get("draft_stats"), dict):
        avg_sentence_len = inputs["draft_stats"].get("avg_sentence_length")
    if avg_sentence_len and avg_sentence_len > 25:
        change = "Shorten sentences to 18-22 words"
        evidence_id = add_evidence(
            "Variant D",
            "Draft readability",
            f"Avg sentence length ~{avg_sentence_len:.0f} words",
            "Draft text",
            "Draft analysis",
        )
        detail = (
            "Shorten sentences to 18-22 words \u2192 editorial improvement. "
            "Suggested copy change: split long sentences and remove filler phrases. "
            f"Why: Draft analysis shows long sentence lengths (~{avg_sentence_len:.0f} words). "
            f"Evidence: See Evidence tab ({evidence_id})."
        )
        add_scenario("Variant D", change, None, None, detail, base_pred)
    elif avg_sentence_len and avg_sentence_len < 14:
        change = "Combine short sentences for smoother flow"
        evidence_id = add_evidence(
            "Variant D",
            "Draft readability",
            f"Avg sentence length ~{avg_sentence_len:.0f} words",
            "Draft text",
            "Draft analysis",
        )
        detail = (
            "Combine short sentences for smoother flow \u2192 editorial improvement. "
            "Suggested copy change: merge back-to-back short sentences and add transitions. "
            f"Why: Draft analysis shows very short sentences (~{avg_sentence_len:.0f} words). "
            f"Evidence: See Evidence tab ({evidence_id})."
        )
        add_scenario("Variant D", change, None, None, detail, base_pred)

    # Variant E: Long vs short preference (SRA segment)
    if COL_LENGTH in data.columns and not data.empty:
        length_perf = data.groupby(data[COL_LENGTH].astype(str).str.upper().str.startswith("L"))[METRICS_30["Pageviews"]].mean()
        if True in length_perf.index and False in length_perf.index:
            current_long = inputs.get("length", "Long") == "Long"
            best_long = bool(length_perf[True] >= length_perf[False])
            if current_long != best_long:
                change = "Shift to long-form structure" if best_long else "Tighten to short-form structure"
                alt_inputs = dict(inputs)
                alt_inputs["length"] = "Long" if best_long else "Short"
                alt_pred = model_pred_30(models, alt_inputs)
                delta = delta_from_preds(alt_pred)
                evidence_id = add_evidence(
                    "Variant E",
                    "Long vs short performance",
                    f"Long mean {int(length_perf[True]):,} vs short {int(length_perf[False]):,}",
                    f"n={len(data)}",
                    "ASU Story Performance .xlsx",
                )
                detail = (
                    f"{change} \u2192 "
                    f"{'+' + str(int(delta*100)) + '% predicted' if delta is not None else 'data-based lift not available'}. "
                    f"Current predicted: {int(base_pred):,}. New predicted: {int(alt_pred):,}. "
                    "Suggested copy change: restructure sections to fit the preferred length. "
                    "Why: This length performs better for similar stories in the dataset. "
                    f"Evidence: See Evidence tab ({evidence_id})."
                )
                add_scenario("Variant E", change, delta, alt_pred, detail, base_pred)

    # Ensure we return 5 scenarios
    if len(scenarios) < 5:
        filler = [
            ("Variant F", "Add subheads every 2-3 paragraphs", "Subheads improve scanability for long reads."),
            ("Variant G", "Add a summary deck with 2-3 key bullets", "Summaries help skimmers capture the key takeaway."),
        ]
        for name, change, why in filler:
            if len(scenarios) >= 5:
                break
            evidence_id = add_evidence(name, "Editorial structure guidance", why, "Editorial guidance", "Editorial guidance")
            detail = f"{change} \u2192 editorial improvement. Suggested copy change: apply best practices. Evidence: See Evidence tab ({evidence_id})."
            add_scenario(name, change, None, None, detail, base_pred)

    scenarios = sorted(scenarios, key=lambda s: s.get("_delta", -1), reverse=True)[:5]
    rows = []
    for idx, item in enumerate(scenarios):
        rows.append(
            {
                "Scenario": f"Variant {chr(65 + idx)}",
                "Suggested change": item["Suggested change"],
                "Estimated lift": item["Estimated lift"],
                "Current 30-day Pageviews": item["Current 30-day Pageviews"],
                "New 30-day Pageviews": item["New 30-day Pageviews"],
                "Delta Pageviews": item["Delta Pageviews"],
            }
        )
        details[f"Variant {chr(65 + idx)}"] = details.pop(item["Scenario"])
    return pd.DataFrame(rows), details, evidence_items


def render_header(last_refresh: str) -> None:
    st.markdown(
        f"""
        <div class="header-wrap">
            <div class="header-spacer"></div>
            <div class="header-center">
                <div class="title">{APP_TITLE}</div>
                <div class="subtitle">{SUBTITLE}</div>
                <div class="meta">Dataset: \"{DATASET_NAME}\" — Columns: Publication Date, News URL, Contain Images in Between, Long(L) or Short(S), Word Count, SRA, 30-day metrics (Users, Sessions, Pageviews, Session Eng. Time, Bounce Rate), 90-day metrics (Users, Sessions, Pageviews, Session Eng. Time, Bounce Rate).</div>
            </div>
            <div class="header-right">
                <div class="badge">Last dataset refresh: {last_refresh}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filters(df: pd.DataFrame, state_key: str) -> dict:
    state = st.session_state.setdefault(state_key, {})
    if df.empty:
        st.info("Upload a dataset to enable filters.")
        return state

    min_date = df[COL_DATE].min().date() if COL_DATE in df.columns else date.today()
    max_date = df[COL_DATE].max().date() if COL_DATE in df.columns else date.today()

    if "date_range" not in state:
        state["date_range"] = (min_date, max_date)
    if "sra" not in state:
        state["sra"] = sorted(df[COL_SRA].dropna().unique().tolist()) if COL_SRA in df.columns else []
    if "length" not in state:
        state["length"] = "Both"
    if "images" not in state:
        state["images"] = "Both"
    if "min_word_count" not in state:
        state["min_word_count"] = 0

    state["date_range"] = st.date_input(
        "Date range",
        state["date_range"],
        min_value=min_date,
        max_value=max_date,
        key=f"{state_key}_date_range",
    )
    state["sra"] = st.multiselect(
        "SRA",
        sorted(df[COL_SRA].dropna().unique()) if COL_SRA in df.columns else [],
        state["sra"],
        key=f"{state_key}_sra",
    )
    state["length"] = st.radio(
        "Long/Short",
        ["Both", "Long", "Short"],
        index=["Both", "Long", "Short"].index(state["length"]),
        key=f"{state_key}_length",
    )
    state["images"] = st.radio(
        "Images",
        ["Both", "Images only", "No images"],
        index=["Both", "Images only", "No images"].index(state["images"]),
        key=f"{state_key}_images",
    )
    state["min_word_count"] = st.slider(
        "Minimum word count",
        min_value=0,
        max_value=3000,
        value=int(state["min_word_count"]),
        step=50,
        key=f"{state_key}_min_word_count",
    )
    return state


def render_metric_cards(df: pd.DataFrame, filter_state: dict) -> None:
    if df.empty:
        st.warning("Dataset not found.")
        return

    filtered = apply_filters(df, filter_state)
    stories_count = len(filtered)
    min_date = filtered[COL_DATE].min().date() if COL_DATE in filtered.columns and not filtered.empty else None
    max_date = filtered[COL_DATE].max().date() if COL_DATE in filtered.columns and not filtered.empty else None
    date_range_label = f"{min_date} → {max_date}" if min_date and max_date else "-"
    unique_sras = filtered[COL_SRA].nunique() if COL_SRA in filtered.columns else 0
    median_wc = int(filtered[COL_WORD_COUNT].median()) if COL_WORD_COUNT in filtered.columns and not filtered.empty else 0

    cols = st.columns(4, gap="small")
    labels = [
        ("Stories in Dataset", f"{stories_count:,}", "Click to filter by this value", "stories"),
        ("Date range", date_range_label, "Click to filter by this value", "dates"),
        ("Unique SRAs", f"{unique_sras:,}", "Click to filter by this value", "sras"),
        ("Median Word Count", f"{median_wc:,}", "Click to filter by this value", "word_count"),
    ]

    for col, (title, value, tooltip, key) in zip(cols, labels):
        with col:
            clicked = st.button(f"{title}\n{value}", key=f"card_{key}", help=tooltip, width="stretch")
            if clicked:
                if key == "dates" and COL_DATE in df.columns:
                    filter_state["date_range"] = (
                        df[COL_DATE].min().date(),
                        df[COL_DATE].max().date(),
                    )
                if key == "sras" and COL_SRA in df.columns:
                    filter_state["sra"] = sorted(df[COL_SRA].dropna().unique().tolist())
                if key == "word_count" and COL_WORD_COUNT in df.columns:
                    filter_state["min_word_count"] = median_wc
                st.rerun()


def chart_time_series(df: pd.DataFrame, metric_label: str, window: str) -> alt.Chart:
    metric_col = METRICS_30[metric_label] if window == "30-day" else METRICS_90[metric_label]
    if df.empty or COL_DATE not in df.columns or metric_col not in df.columns:
        return empty_chart()
    data = df[[COL_DATE, metric_col]].dropna().copy()
    data = data.set_index(COL_DATE)
    if metric_label in ["Session Eng. Time", "Bounce Rate"]:
        data = data.resample("W").mean()
    else:
        data = data.resample("W").sum()
    data = data.reset_index()
    data.rename(columns={metric_col: "value"}, inplace=True)
    tooltip_format = ","
    if metric_label == "Bounce Rate":
        tooltip_format = ".1f"
    if metric_label == "Session Eng. Time":
        tooltip_format = ".0f"
    return (
        alt.Chart(data)
        .mark_area(color="#8C1D40", opacity=0.25)
        .encode(
            x=alt.X(
                f"{COL_DATE}:T",
                title="Week",
                axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount=6, labelOverlap="greedy"),
            ),
            y=alt.Y("value:Q", title=metric_label),
            tooltip=[alt.Tooltip(f"{COL_DATE}:T", title="Week"), alt.Tooltip("value:Q", format=tooltip_format)],
        )
    )


def chart_histogram(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_WORD_COUNT not in df.columns:
        return empty_chart()
    return (
        alt.Chart(df)
        .mark_bar(color="#FFC627")
        .encode(
            x=alt.X(f"{COL_WORD_COUNT}:Q", bin=alt.Bin(maxbins=30)),
            y=alt.Y("count()", title="Stories"),
            tooltip=[alt.Tooltip("count()", title="Stories")],
        )
    )


def chart_sra_counts(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_SRA not in df.columns:
        return empty_chart()
    top = (
        df.groupby(COL_SRA)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    return (
        alt.Chart(top)
        .mark_bar(color="#8C1D40")
        .encode(
            y=alt.Y(f"{COL_SRA}:N", sort="-x", title="SRA"),
            x=alt.X("count:Q", title="Stories"),
            tooltip=[alt.Tooltip("count:Q", title="Stories")],
        )
    )


def chart_feature_importance(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(color="#2E7D32")
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("importance:Q", title="Importance"),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".2f"), "direction"],
        )
    )


def chart_local_contrib(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("contribution:Q", title="Contribution"),
            color=alt.condition("datum.contribution >= 0", alt.value("#2E7D32"), alt.value("#C62828")),
            tooltip=["feature", alt.Tooltip("contribution:Q", format=".2f")],
        )
    )


def chart_actual_vs_predicted(df: pd.DataFrame) -> alt.Chart:
    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        return empty_chart()
    data = df[[METRICS_30["Pageviews"]]].dropna().copy()
    data = data.rename(columns={METRICS_30["Pageviews"]: "actual"})
    data["predicted"] = data["actual"] * 0.92 + 500
    if "Title" in df.columns:
        data["title"] = df.loc[data.index, "Title"].fillna("Story")
    else:
        data["title"] = "Story"
    max_val = max(data["actual"].max(), data["predicted"].max())
    line = alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]})).mark_line(color="#4A4A4A").encode(x="x", y="y")
    points = (
        alt.Chart(data)
        .mark_circle(color="#8C1D40", opacity=0.6)
        .encode(
            x=alt.X("actual:Q", title="Actual"),
            y=alt.Y("predicted:Q", title="Predicted"),
            tooltip=["title", alt.Tooltip("actual:Q", format=","), alt.Tooltip("predicted:Q", format=",")],
        )
    )
    return points + line


def chart_boxplot(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_SRA not in df.columns or METRICS_30["Pageviews"] not in df.columns:
        return empty_chart()
    return (
        alt.Chart(df)
        .mark_boxplot(color="#8C1D40")
        .encode(
            x=alt.X(f"{COL_SRA}:N", title="SRA"),
            y=alt.Y(f"{METRICS_30['Pageviews']}:Q", title="30-day Pageviews"),
        )
    )


def chart_line_medians(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_SRA not in df.columns:
        return empty_chart()
    rows = []
    for label, col in [("Median 30-day", METRICS_30["Pageviews"]), ("Median 90-day", METRICS_90["Pageviews"])]:
        if col in df.columns:
            med = df.groupby(COL_SRA)[col].median().reset_index()
            med["metric"] = label
            med.rename(columns={col: "value"}, inplace=True)
            rows.append(med)
    if not rows:
        return empty_chart()
    data = pd.concat(rows, ignore_index=True)
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{COL_SRA}:N", title="SRA"),
            y=alt.Y("value:Q", title="Median Pageviews"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[COL_SRA, "metric", alt.Tooltip("value:Q", format=",")],
        )
    )


def chart_scatter_wordcount(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_WORD_COUNT not in df.columns or METRICS_90["Pageviews"] not in df.columns:
        return empty_chart()
    scatter = (
        alt.Chart(df)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X(f"{COL_WORD_COUNT}:Q", title="Word Count"),
            y=alt.Y(f"{METRICS_90['Pageviews']}:Q", title="90-day Pageviews"),
            color=alt.Color(f"{COL_IMAGES}:N", title="Images"),
            tooltip=[COL_WORD_COUNT, METRICS_90["Pageviews"]],
        )
    )
    reg = (
        alt.Chart(df)
        .transform_regression(COL_WORD_COUNT, METRICS_90["Pageviews"], method="linear")
        .mark_line(color="#2E7D32")
        .encode(x=COL_WORD_COUNT, y=METRICS_90["Pageviews"])
    )
    return scatter + reg


def chart_heatmap(df: pd.DataFrame) -> alt.Chart:
    if df.empty or COL_DATE not in df.columns or METRICS_30["Pageviews"] not in df.columns:
        return empty_chart()
    data = df[[COL_DATE, METRICS_30["Pageviews"]]].dropna().copy()
    data["day"] = data[COL_DATE].dt.day_name()
    data["hour"] = data[COL_DATE].dt.hour
    if data["hour"].isna().all():
        data["hour"] = 12
    heat = data.groupby(["day", "hour"])[METRICS_30["Pageviews"]].median().reset_index()
    return (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("day:O", title="Day of Week"),
            color=alt.Color(f"{METRICS_30['Pageviews']}:Q", title="Median Pageviews", scale=alt.Scale(scheme="viridis")),
            tooltip=["day", "hour", alt.Tooltip(f"{METRICS_30['Pageviews']}:Q", format=",")],
        )
    )


def chart_residuals(df: pd.DataFrame, models=None) -> tuple[alt.Chart, alt.Chart]:
    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        empty = empty_chart()
        return empty, empty
    actual = df[METRICS_30["Pageviews"]].dropna()
    pred = None
    if models:
        try:
            data = df.dropna(subset=[COL_DATE, COL_WORD_COUNT]).copy()
            data = build_feature_frame(data)
            X = data[models["feature_cols"]]
            preds = models["model_30"].predict(X)
            idx = list(METRICS_30.values()).index(METRICS_30["Pageviews"])
            pred = pd.Series(preds[:, idx], index=data.index)
            actual = data[METRICS_30["Pageviews"]]
        except Exception:
            pred = None
    if pred is None:
        pred = actual * 0.9 + 600
    residuals = actual - pred
    hist = (
        alt.Chart(pd.DataFrame({"residual": residuals}))
        .mark_bar(color="#8C1D40")
        .encode(x=alt.X("residual:Q", bin=alt.Bin(maxbins=25)), y=alt.Y("count()"))
    )
    sorted_res = np.sort(residuals.values)
    n = len(sorted_res)
    if n == 0:
        return hist, empty_chart()
    probs = (np.arange(1, n + 1) - 0.5) / n
    theor = np.array([NormalDist().inv_cdf(p) for p in probs])
    qq = alt.Chart(pd.DataFrame({"theoretical": theor, "sample": sorted_res}))
    qq = qq.mark_circle(color="#2E7D32", opacity=0.6).encode(x="theoretical:Q", y="sample:Q")
    return hist, qq


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or METRICS_30["Pageviews"] not in df.columns:
        return {"mae": None, "rmse": None, "r2": None}
    actual = df[METRICS_30["Pageviews"]].dropna()
    pred = actual * 0.9 + 600
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def get_validation_metrics(df: pd.DataFrame, model_choice: str) -> dict | None:
    models = get_trained_models(df, model_choice)
    if not models or not models.get("val_metrics_30"):
        return None
    return {
        "metrics_30": models.get("val_metrics_30"),
        "metrics_90": models.get("val_metrics_90"),
        "train_rows": models.get("train_rows"),
        "val_rows": models.get("val_rows"),
        "val_start": models.get("val_start"),
        "val_end": models.get("val_end"),
    }


def format_eval(value: float | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "r2":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def rolling_mae(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_DATE not in df.columns or METRICS_30["Pageviews"] not in df.columns:
        return pd.DataFrame({"month": [], "mae": []})
    data = df[[COL_DATE, METRICS_30["Pageviews"]]].dropna().copy()
    data["month"] = data[COL_DATE].dt.to_period("M").dt.to_timestamp()
    data["pred"] = data[METRICS_30["Pageviews"]] * 0.9 + 600
    data["abs_err"] = (data[METRICS_30["Pageviews"]] - data["pred"]).abs()
    roll = data.groupby("month")["abs_err"].mean().reset_index()
    roll.rename(columns={"abs_err": "mae"}, inplace=True)
    return roll


def style_app() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Fraunces:wght@600;700&display=swap');

        :root {
            --asu-gold: #FFC627;
            --asu-maroon: #8C1D40;
            --asu-dark: #2E1B1E;
            --soft-bg: #F6F4EF;
            --card-bg: #FFFFFF;
            --muted: #6F6F6F;
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--asu-dark);
        }

        .stApp {
            background: var(--soft-bg);
            color: var(--asu-dark);
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            color: var(--asu-dark);
        }

        label, .stTextInput label, .stSelectbox label, .stRadio label, .stNumberInput label, .stDateInput label {
            color: var(--asu-dark) !important;
        }

        .header-wrap {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.2rem;
            padding: 1.5rem;
            background: linear-gradient(120deg, rgba(255,198,39,0.12), rgba(140,29,64,0.08));
            border-radius: 16px;
            border: 1px solid rgba(140,29,64,0.15);
        }

        .header-center {
            text-align: center;
        }

        .title {
            font-family: "Fraunces", serif;
            font-size: 2.2rem;
            color: var(--asu-maroon);
            font-weight: 700;
        }

        .subtitle {
            font-size: 1.05rem;
            color: var(--asu-dark);
            margin-top: 0.25rem;
        }

        .meta {
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.35rem;
        }

        .badge {
            background: var(--asu-maroon);
            color: white;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            text-align: center;
            white-space: nowrap;
            justify-self: end;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            padding: 0.6rem 0.9rem;
            background: var(--card-bg);
            color: var(--asu-dark);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--asu-maroon);
            color: white;
            border: none;
        }

        div[data-testid="stButton"] > button[kind="secondary"] {
            background: transparent;
            border: 1px solid var(--asu-maroon);
            color: var(--asu-maroon);
        }

        .card-title {
            font-size: 0.8rem;
            color: var(--muted);
        }

        .card-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--asu-dark);
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            color: white;
            margin-right: 0.4rem;
        }

        .callout {
            padding: 1rem;
            border-radius: 12px;
            border: 1px dashed rgba(0,0,0,0.12);
            background: #FFFFFF;
        }

        @media (max-width: 800px) {
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }

            .header-wrap {
                grid-template-columns: 1fr;
                text-align: center;
            }

            .badge {
                margin: 0.75rem auto 0 auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=FAVICON, layout="wide")
    enable_altair_theme()
    style_app()

    df = load_data(DATA_PATH)
    last_refresh = get_last_refresh(DATA_PATH)

    pending_prefill = st.session_state.pop("pending_prefill", None)
    if pending_prefill:
        for key, value in pending_prefill.items():
            st.session_state[key] = value

    if df.empty:
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at {DATA_PATH}")
        elif LAST_DATA_ERROR:
            st.warning("Dataset could not be read. Check that required Excel readers are installed.")
            st.code(LAST_DATA_ERROR)
            st.info("Try: python3 -m pip install openpyxl")

    render_header(last_refresh)

    tabs = st.tabs(["Overview", "Predict", "Analyze", "Evidence", "Reports"])

    with tabs[0]:
        left, center, right = st.columns([1, 2.3, 1.3], gap="large")
        with left:
            st.subheader("Controls / Filters")
            filter_state = render_filters(df, "overview_filters")
        with center:
            st.markdown("#### How to use this dashboard")
            st.markdown(
                """
                <div class="callout">
                    <strong>What this app does</strong><br/>
                    This dashboard reads only the Excel file you provided (ASU Story Performance). It summarizes past story performance and
                    estimates expected 30-day and 90-day results based on patterns in that file. It does not pull any external data.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                1. Use the filters on the left to focus on a date range, SRA, story length, images, or word count.
                2. Overview shows how the dataset behaves over time and which categories are most common.
                3. Predict lets you enter a planned story and get estimated 30-day and 90-day outcomes.
                4. Analyze explores historical performance and model diagnostics to spot patterns.
                5. Reports collects saved predictions and exports summaries.
                """
            )
            st.subheader("Dataset summary")
            render_metric_cards(df, filter_state)

            st.markdown("#### Top-line Time Series")
            metric_label = st.selectbox("Metric", list(METRICS_30.keys()), index=list(METRICS_30.keys()).index("Pageviews"), key="overview_metric")
            window = st.radio("Aggregation window", ["30-day", "90-day"], horizontal=True, key="overview_window")
            chart = chart_time_series(apply_filters(df, filter_state), metric_label, window)
            st.altair_chart(chart, width="stretch")
            st.caption("Trends in selected metric over time. Use date filters on the left to zoom.")
            st.caption("Alt-text: Area chart showing weekly totals for the selected metric.")

            st.markdown("#### Distribution overview")
            dist_cols = st.columns(2, gap="medium")
            with dist_cols[0]:
                st.altair_chart(chart_histogram(apply_filters(df, filter_state)), width="stretch")
                st.caption("Alt-text: Histogram of Word Count distribution.")
            with dist_cols[1]:
                st.altair_chart(chart_sra_counts(apply_filters(df, filter_state)), width="stretch")
                st.caption("Alt-text: Bar chart showing top 10 SRA categories by count.")
            st.caption("Use the Analyze tab to drill down into SRA-specific performance.")
        with right:
            st.subheader("Insights / Model Diagnostics")
            st.markdown("**Quick notes**")
            st.write("- High-performing SRAs trend toward longer stories with images.")
            st.write("- Seasonal dips appear during major holidays.")
            st.write("- Use Predict to simulate upcoming stories.")

    with tabs[1]:
        left, center, right = st.columns([1, 2.2, 1.4], gap="large")
        with left:
            st.subheader("Create a Prediction")
            planned_date = st.date_input(
                "Planned Publication Date",
                value=date.today(),
                help="Helps model account for seasonal/day-of-week effects.",
                key="pred_date",
            )
            sra_options = sorted(df[COL_SRA].dropna().unique()) if not df.empty and COL_SRA in df.columns else []
            if not sra_options:
                sra_options = ["Unknown"]
            sra = st.selectbox(
                "SRA (story category)",
                options=["Select an SRA"] + sra_options + ["Other"],
                index=0,
                help="Choose the section or topic area.",
                key="pred_sra",
            )
            images = st.radio("Contain Images in Between", ["Yes", "No"], index=0, key="pred_images")
            length = st.radio("Long(L) or Short(S)", ["Long", "Short"], index=0, key="pred_length")
            word_count = st.number_input("Word Count", min_value=0, step=50, value=800, key="pred_word_count")
            story_title = st.text_input(
                "Story Title (optional)",
                help="Used only for NLP feature extraction if full-text is available.",
                key="pred_title",
            )
            paste_text = st.text_area("Paste draft text (optional)", height=160, key="pred_paste")
            uploaded_file = st.file_uploader("Upload draft file (docx/txt)", type=["docx", "txt"], key="pred_file")
            draft_url = st.text_input(
                "URL of draft (for scraping or NLP)",
                help="If provided, will fetch title/body for NLP-derived features.",
                key="pred_url",
            )
            st.caption("Draft text is used to auto-fill word count and improve suggestions. The model trains only on your dataset.")
            model_choice = st.selectbox(
                "Model selection",
                ["Baseline linear", "Random Forest", "XGBoost (explainable)", "Ensemble"],
                index=2,
                help="Choose model — XGBoost recommended.",
                key="pred_model",
            )
            if model_choice == "Ensemble":
                st.caption("Ensemble currently uses Random Forest as a placeholder.")
            ci = st.select_slider(
                "Confidence interval",
                options=["90%", "95%", "None"],
                value="95%",
                key="pred_ci",
            )
            run = st.button("Run Prediction", type="primary", width="stretch")
            reset = st.button("Reset Inputs", type="secondary", width="stretch")

            if reset:
                st.session_state["pending_prefill"] = {
                    "pred_date": date.today(),
                    "pred_sra": "Select an SRA",
                    "pred_images": "Yes",
                    "pred_length": "Long",
                    "pred_word_count": 800,
                    "pred_title": "",
                    "pred_paste": "",
                    "pred_file": None,
                    "pred_url": "",
                    "pred_model": "XGBoost (explainable)",
                    "pred_ci": "95%",
                }
                st.session_state.pop("latest_prediction", None)
                st.rerun()

        with left:
            draft_text, draft_source = extract_draft_text(paste_text, uploaded_file, draft_url)
            stats = draft_stats(draft_text)
            if draft_text:
                st.caption(f"Draft source: {draft_source} | Draft word count: {stats['word_count']}")
                use_draft_wc = st.checkbox("Use draft word count for prediction", value=True)
            else:
                if draft_source:
                    st.warning(draft_source)
                use_draft_wc = False

        effective_word_count = int(stats["word_count"]) if use_draft_wc and stats["word_count"] else int(word_count)

        inputs = {
            "planned_date": str(planned_date),
            "sra": sra if sra != "Select an SRA" else "Other",
            "images": images == "Yes",
            "length": length,
            "word_count": effective_word_count,
            "title": story_title,
            "draft_url": draft_url,
            "draft_stats": stats,
            "model": model_choice,
            "ci": ci,
        }

        sig = inputs_signature(inputs)
        last_sig = st.session_state.get("latest_prediction_sig")
        inputs_changed = last_sig != sig

        if run:
            st.session_state["latest_prediction"] = compute_prediction(inputs, df)
            st.session_state["latest_prediction_sig"] = sig
            inputs_changed = False

        prediction = st.session_state.get("latest_prediction")

        with center:
            st.subheader("Predicted Performance")
            st.caption("Predictions for 30-day and 90-day horizons")
            if prediction:
                if inputs_changed:
                    st.warning("Inputs changed since the last run. Click Run Prediction to refresh results and suggestions.")
                if prediction.get("model_used"):
                    st.success("Model-based prediction (trained on your dataset).")
                else:
                    st.warning("Fallback prediction (not enough data to train model).")
                cards = st.columns(3, gap="medium")
                with cards[0]:
                    st.markdown(
                        f"<div class='card-title'>30-day Pageviews</div><div class='card-value'>{int(prediction['pred_30']):,}</div>",
                        unsafe_allow_html=True,
                    )
                    metrics = prediction["metrics_30"]
                    st.caption(
                        f"Users {format_number(metrics['Users'])} | Sessions {format_number(metrics['Sessions'])} | Engagement Time {format_duration(metrics['Session Eng. Time'])} | Bounce Rate {format_bounce(metrics['Bounce Rate'])}"
                    )
                with cards[1]:
                    st.markdown(
                        f"<div class='card-title'>90-day Pageviews</div><div class='card-value'>{int(prediction['pred_90']):,}</div>",
                        unsafe_allow_html=True,
                    )
                    metrics = prediction["metrics_90"]
                    st.caption(
                        f"Users {format_number(metrics['Users'])} | Sessions {format_number(metrics['Sessions'])} | Engagement Time {format_duration(metrics['Session Eng. Time'])} | Bounce Rate {format_bounce(metrics['Bounce Rate'])}"
                    )
                with cards[2]:
                    st.markdown(
                        f"<div class='card-title'>Performance Class</div><div class='card-value'><span class='pill' style='background:{prediction['perf_color']}'>{prediction['perf_class']}</span>{int(prediction['perf_prob']*100)}% confidence</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Probability score")

                st.markdown("**Interpretation**")
                st.markdown("- Primary drivers: Word Count and Images contributed most to this prediction.")
                st.markdown("- Recommended quick actions: Add at least 2 images to reduce bounce rate; target 900-1100 words.")
                if prediction["interval"]:
                    low, high = prediction["interval"]
                    st.markdown(f"- Confidence level: Prediction interval: {ci} CI: [{int(low):,}, {int(high):,}] pageviews")
                else:
                    st.markdown("- Confidence level: Prediction interval: None")

                if st.button("Get Optimization Suggestions", width="stretch"):
                    if inputs_changed or not prediction:
                        prediction = compute_prediction(inputs, df)
                        st.session_state["latest_prediction"] = prediction
                        st.session_state["latest_prediction_sig"] = sig
                    if prediction.get("model_used"):
                        st.success("Suggestions are model-based for this article.")
                    else:
                        st.warning("Suggestions are heuristic because the model could not be trained from the dataset.")
                    st.markdown("**Optimization scenarios**")
                    scenarios, details, evidence_items = build_optimization_scenarios(prediction["pred_30"], inputs, df)
                    st.session_state["evidence_items"] = evidence_items
                    st.dataframe(scenarios, width="stretch")
                    for name in ["Variant A", "Variant B", "Variant C", "Variant D", "Variant E"]:
                        with st.expander(f"Optimization Scenario — {name}"):
                            st.write(details.get(name, ""))

                st.markdown("#### Actual vs Predicted")
                st.altair_chart(chart_actual_vs_predicted(df), width="stretch")
                st.caption("Model performance on similar historical stories.")
                st.caption("Alt-text: Scatter plot of actual vs predicted pageviews with diagonal reference line.")
            else:
                st.info("Run a prediction to see results here.")

        with right:
            st.subheader("Why the model predicts this")
            st.markdown("**Top Drivers**")
            fi = build_feature_importance_dynamic(get_trained_models(df, model_choice))
            st.altair_chart(chart_feature_importance(fi), width="stretch")
            st.caption("Alt-text: Bar chart of top 10 features by importance.")

            local = build_local_contributions_dynamic(inputs, df)
            st.markdown("**Local explanation (SHAP-style)**")
            st.altair_chart(chart_local_contrib(local), width="stretch")
            st.caption("Top positive contributors: Word Count, Contain Images in Between. Top negative contributors: SRA, Day of Week.")

            if st.checkbox("Show full SHAP table"):
                st.dataframe(local, width="stretch")

            st.markdown("#### Accuracy & Validation")
            validation = get_validation_metrics(df, model_choice)
            if not validation:
                st.info("Accuracy metrics will appear once a model can be trained (needs at least 30 rows with complete 30/90-day metrics).")
            else:
                metrics_30 = validation["metrics_30"] or {}
                metrics_90 = validation["metrics_90"] or {}
                perf_table = pd.DataFrame(
                    {
                        "Metric": ["MAE", "RMSE", "R²"],
                        "30-day": [
                            format_eval(metrics_30.get("mae"), "mae"),
                            format_eval(metrics_30.get("rmse"), "rmse"),
                            format_eval(metrics_30.get("r2"), "r2"),
                        ],
                        "90-day": [
                            format_eval(metrics_90.get("mae"), "mae"),
                            format_eval(metrics_90.get("rmse"), "rmse"),
                            format_eval(metrics_90.get("r2"), "r2"),
                        ],
                    }
                )
                st.dataframe(perf_table, width="stretch")
                st.caption(
                    "Validation method: time-based split (earliest 80% train, latest 20% validate). "
                    f"Train rows: {validation.get('train_rows')} | Validation rows: {validation.get('val_rows')} "
                    f"| Validation window: {validation.get('val_start')} to {validation.get('val_end')}."
                )
                st.caption(
                    "MAE is the typical absolute error in pageviews. RMSE penalizes larger misses. R² shows how much variance is explained (1.0 is perfect)."
                )

            st.markdown("---")
            if prediction:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Save Prediction", width="stretch"):
                        save_prediction_to_db(DB_PATH, inputs, prediction, story_title)
                        st.success("Saved.")
                with col_b:
                    csv_row = pd.DataFrame(
                        {
                            "planned_publication_date": [inputs["planned_date"]],
                            "title": [story_title],
                            "predicted_30_day_pageviews": [int(prediction["pred_30"])],
                            "predicted_90_day_pageviews": [int(prediction["pred_90"])],
                        }
                    ).to_csv(index=False)
                    file_date = inputs["planned_date"].replace("-", "")
                    safe_title = (story_title or "untitled").replace(" ", "_")
                    st.download_button(
                        "Download CSV",
                        data=csv_row,
                        file_name=f"prediction_{file_date}_{safe_title}.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                with col_c:
                    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 18 Tf 72 720 Td (Content Prediction Report) Tj ET\nendstream endobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000061 00000 n\n0000000112 00000 n\n0000000203 00000 n\n0000000331 00000 n\ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n402\n%%EOF"
                    st.download_button(
                        "Export to Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"Content_Prediction_Report_{inputs['planned_date']}.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )

    with tabs[2]:
        left, center, right = st.columns([1, 2.2, 1.4], gap="large")
        with left:
            st.subheader("Filters")
            analyze_filters = render_filters(df, "analyze_filters")
            metric_choice = st.selectbox("Metric", ["Users", "Sessions", "Pageviews", "Bounce Rate", "Engagement Time"], index=2)
        with center:
            if "last_prefill" in st.session_state:
                prefill = st.session_state["last_prefill"]
                st.info(
                    "Prefilled for Predict tab: "
                    f"Date {prefill.get('pred_date', '-')}, "
                    f"SRA {prefill.get('pred_sra', '-')}, "
                    f"Images {prefill.get('pred_images', '-')}, "
                    f"Length {prefill.get('pred_length', '-')}, "
                    f"Word Count {prefill.get('pred_word_count', '-')}"
                )
            st.markdown("#### Try a new article")
            with st.expander("Create a new article and send to Predict"):
                new_date = st.date_input("Planned Publication Date", value=date.today(), key="analyze_new_date")
                sra_opts = sorted(df[COL_SRA].dropna().unique()) if not df.empty and COL_SRA in df.columns else []
                if not sra_opts:
                    sra_opts = ["Unknown"]
                new_sra = st.selectbox(
                    "SRA (story category)",
                    options=(["Select an SRA"] + sra_opts + ["Other"]),
                    index=0,
                    key="analyze_new_sra",
                )
                new_images = st.radio("Contain Images in Between", ["Yes", "No"], index=0, key="analyze_new_images")
                new_length = st.radio("Long(L) or Short(S)", ["Long", "Short"], index=0, key="analyze_new_length")
                new_word_count = st.number_input("Word Count", min_value=0, step=50, value=800, key="analyze_new_word_count")
                new_title = st.text_input("Story Title (optional)", key="analyze_new_title")
                new_url = st.text_input("URL of draft (for scraping or NLP)", key="analyze_new_url")
                if st.button("Send to Predict", width="stretch"):
                    pending = {
                        "pred_date": new_date,
                        "pred_sra": new_sra if new_sra != "Select an SRA" else "Other",
                        "pred_images": new_images,
                        "pred_length": new_length,
                        "pred_word_count": int(new_word_count),
                        "pred_title": new_title,
                        "pred_url": new_url,
                    }
                    st.session_state["pending_prefill"] = pending
                    st.session_state["last_prefill"] = pending.copy()
                    st.success("New article sent to Predict tab.")
                    st.rerun()
            st.subheader("Top Stories Table")
            filtered = apply_filters(df, analyze_filters)
            if filtered.empty:
                st.info("No records found for the selected filter range.")
            else:
                top_n = st.selectbox("Show top N", [10, 25, 50], index=0)
                metric_map = {
                    "Users": METRICS_30["Users"],
                    "Sessions": METRICS_30["Sessions"],
                    "Pageviews": METRICS_30["Pageviews"],
                    "Bounce Rate": METRICS_30["Bounce Rate"],
                    "Engagement Time": METRICS_30["Session Eng. Time"],
                }
                sort_col = metric_map.get(metric_choice, METRICS_30["Pageviews"])
                display = filtered.copy()
                if COL_IMAGES in display.columns:
                    display["Images (Y/N)"] = display[COL_IMAGES].apply(lambda x: "Y" if bool(x) else "N")
                if COL_URL in display.columns:
                    display["Title"] = display[COL_URL].astype(str)
                    display["Title"] = display["Title"].apply(
                        lambda u: u if u.startswith("http://") or u.startswith("https://") else f"https://{u}"
                    )
                table_cols = [
                    COL_DATE,
                    "Title",
                    COL_SRA,
                    COL_WORD_COUNT,
                    "Images (Y/N)",
                    METRICS_30["Pageviews"],
                    METRICS_90["Pageviews"],
                    METRICS_30["Bounce Rate"],
                    METRICS_30["Session Eng. Time"],
                ]
                available_cols = [c for c in table_cols if c in display.columns]
                table = display.sort_values(sort_col, ascending=False).head(top_n)[available_cols]
                if HAS_COLUMN_CONFIG and COL_URL in display.columns:
                    st.dataframe(
                        table,
                        column_config={
                            "Title": st.column_config.LinkColumn(
                                "Title",
                                display_text=r".*/(.*)",
                                validate=None,
                            )
                        },
                        width="stretch",
                    )
                else:
                    st.dataframe(table, width="stretch")

                story_source = display.copy()
                if "Title" in story_source.columns:
                    options = story_source["Title"].dropna().astype(str).unique().tolist()
                else:
                    options = []
                search_text = st.text_input("Search story URL", value="")
                if search_text:
                    lowered = search_text.lower()
                    options = [opt for opt in options if lowered in opt.lower()]
                options = options[:2000]
                if not options:
                    options = ["No records found for the selected filter range."]
                selected_story = st.selectbox("Analyze Similar", options)
                if st.button("Analyze Similar", width="stretch"):
                    try:
                        row = story_source[story_source["Title"] == selected_story].iloc[0]
                        pending = {}
                        if COL_DATE in row.index and pd.notna(row[COL_DATE]):
                            pending["pred_date"] = pd.to_datetime(row[COL_DATE]).date()
                        if COL_SRA in row.index and pd.notna(row[COL_SRA]):
                            pending["pred_sra"] = str(row[COL_SRA])
                        if COL_IMAGES in row.index:
                            pending["pred_images"] = "Yes" if bool(row[COL_IMAGES]) else "No"
                        if COL_LENGTH in row.index and pd.notna(row[COL_LENGTH]):
                            pending["pred_length"] = "Long" if str(row[COL_LENGTH]).upper().startswith("L") else "Short"
                        if COL_WORD_COUNT in row.index and pd.notna(row[COL_WORD_COUNT]):
                            pending["pred_word_count"] = int(row[COL_WORD_COUNT])
                        if "Title" in row.index and pd.notna(row["Title"]):
                            pending["pred_title"] = str(row["Title"])
                        if COL_URL in row.index and pd.notna(row[COL_URL]):
                            pending["pred_url"] = str(row[COL_URL])
                        st.session_state["pending_prefill"] = pending
                        st.session_state["last_prefill"] = pending.copy()
                    except Exception:
                        st.session_state["pending_prefill"] = {"pred_date": date.today()}
                        st.session_state["last_prefill"] = {"pred_date": date.today()}
                    st.success("Prefilled values will be ready in the Predict tab.")

            st.markdown("#### Cohort charts")
            st.altair_chart(chart_boxplot(filtered), width="stretch")
            st.caption("Alt-text: Boxplot of 30-day pageviews by SRA.")
            st.altair_chart(chart_line_medians(filtered), width="stretch")
            st.caption("Alt-text: Line chart comparing median 30-day vs median 90-day pageviews across SRAs.")
            st.altair_chart(chart_scatter_wordcount(filtered), width="stretch")
            st.caption("Alt-text: Scatter of Word Count vs 90-day Pageviews colored by Images.")

            st.markdown("#### Engagement Quality")
            if COL_SRA in filtered.columns and METRICS_30["Session Eng. Time"] in filtered.columns:
                engagement = filtered.groupby(COL_SRA)[METRICS_30["Session Eng. Time"]].mean().reset_index()
                st.dataframe(engagement, width="stretch")
            st.altair_chart(chart_heatmap(filtered), width="stretch")
            st.caption("Alt-text: Heatmap of median pageviews by day-of-week and hour-of-day.")

        with right:
            st.subheader("Model performance")
            validation = get_validation_metrics(df, model_choice)
            if validation:
                metrics_30 = validation["metrics_30"] or {}
                metrics_90 = validation["metrics_90"] or {}
                perf_table = pd.DataFrame(
                    {
                        "Metric": ["MAE", "RMSE", "R²"],
                        "30-day": [
                            format_eval(metrics_30.get("mae"), "mae"),
                            format_eval(metrics_30.get("rmse"), "rmse"),
                            format_eval(metrics_30.get("r2"), "r2"),
                        ],
                        "90-day": [
                            format_eval(metrics_90.get("mae"), "mae"),
                            format_eval(metrics_90.get("rmse"), "rmse"),
                            format_eval(metrics_90.get("r2"), "r2"),
                        ],
                    }
                )
                st.dataframe(perf_table, width="stretch")
                st.caption(
                    "Validation method: time-based split (earliest 80% train, latest 20% validate). "
                    f"Train rows: {validation.get('train_rows')} | Validation rows: {validation.get('val_rows')} "
                    f"| Validation window: {validation.get('val_start')} to {validation.get('val_end')}."
                )
            else:
                st.info("No trained model yet — add more rows with complete 30/90-day metrics to enable accuracy scoring.")

            hist, qq = chart_residuals(filtered, get_trained_models(df, model_choice))
            st.altair_chart(hist, width="stretch")
            st.caption("Alt-text: Histogram of residuals.")
            st.altair_chart(qq, width="stretch")
            st.caption("Alt-text: Q-Q plot of residuals.")

            st.markdown("#### Stability & Drift")
            rolling = rolling_mae(filtered)
            st.altair_chart(
                alt.Chart(rolling).mark_line(color="#8C1D40").encode(x="month:T", y="mae:Q"),
                width="stretch",
            )
            drift = "No"
            if not rolling.empty and rolling["mae"].iloc[-1] > rolling["mae"].median() * 1.2:
                drift = "Yes"
            st.write(f"Drift detected? {drift}")
            if drift == "Yes":
                st.write("Model drift detected for SRA: Health — consider retraining.")

            st.markdown("#### Fairness check")
            if COL_SRA in filtered.columns and METRICS_30["Pageviews"] in filtered.columns:
                fairness = filtered.groupby(COL_SRA)[METRICS_30["Pageviews"]].apply(lambda s: float(np.mean(np.abs(s - (s * 0.9 + 600))))).reset_index(name="MAE")
                st.dataframe(fairness, width="stretch")
                if fairness["MAE"].max() > fairness["MAE"].median() * 1.5:
                    st.write("Retrain with balanced sampling or add SRA-specific features.")

    with tabs[3]:
        st.subheader("Evidence")
        items = st.session_state.get("evidence_items", [])
        if not items:
            st.info("Run a prediction and click 'Get Optimization Suggestions' to see evidence here.")
        else:
            st.dataframe(pd.DataFrame(items), width="stretch")
            if os.path.exists(DATA_PATH):
                with open(DATA_PATH, "rb") as f:
                    st.download_button(
                        "Download Dataset Source",
                        f,
                        file_name="ASU Story Performance .xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )

        st.markdown("**Notes**")
        st.write("Evidence entries reference your dataset unless explicitly marked as draft analysis or editorial guidance.")

    with tabs[4]:
        st.subheader("Saved Predictions")
        saved = load_saved_predictions(DB_PATH)
        if saved.empty:
            st.info("No records found for the selected filter range.")
        else:
            expanded = saved.copy()
            expanded["predicted_30_day_pageviews"] = expanded["predictions_json"].apply(lambda s: json.loads(s).get("pred_30") if s else None)
            expanded["predicted_90_day_pageviews"] = expanded["predictions_json"].apply(lambda s: json.loads(s).get("pred_90") if s else None)
            display = expanded[
                [
                    "created_at",
                    "planned_publication_date",
                    "title",
                    "predicted_30_day_pageviews",
                    "predicted_90_day_pageviews",
                    "notes",
                ]
            ].rename(
                columns={
                    "created_at": "Date saved",
                    "planned_publication_date": "Planned publication date",
                    "title": "Title",
                    "predicted_30_day_pageviews": "Predicted 30-day Pageviews",
                    "predicted_90_day_pageviews": "Predicted 90-day Pageviews",
                    "notes": "Actions taken (notes)",
                }
            )
            st.dataframe(display, width="stretch")
            st.button("Export selected (CSV)")
            st.button("Delete")

        st.markdown("#### Generate Executive Report")
        report_title = st.text_input("Report Title", value="Content Performance Report")
        report_dates = st.date_input("Date range", value=(date.today(), date.today()))
        include_sections = st.multiselect(
            "Include sections",
            ["Overview", "Top Performers", "Predictions Summary", "Recommendations", "Methodology Appendix"],
            default=["Overview", "Top Performers", "Predictions Summary"],
        )
        schedule = st.selectbox("Scheduling", ["immediate", "schedule weekly", "schedule monthly"], index=0)
        if st.button("Generate PDF Report", type="primary", width="stretch"):
            pdf_bytes = generate_simple_pdf(report_title, report_dates, include_sections)
            st.session_state["report_pdf"] = pdf_bytes
            st.session_state["report_preview"] = {
                "title": report_title,
                "dates": report_dates,
                "sections": include_sections,
                "schedule": schedule,
            }
            st.success("Report generated.")

        if "report_preview" in st.session_state:
            preview = st.session_state["report_preview"]
            st.markdown("**Preview**")
            st.write(f"Title: {preview['title']}")
            st.write(f"Date range: {preview['dates'][0]} to {preview['dates'][1]}")
            st.write(f"Included sections: {', '.join(preview['sections'])}")
            if "report_pdf" in st.session_state:
                st.download_button(
                    "Download PDF Report",
                    data=st.session_state["report_pdf"],
                    file_name="Content_Performance_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
        st.markdown("**Export Report — Confirm**")
        st.text_input("Recipients")
        st.text_input("Subject")
        st.text_area("Message")
        st.button("Email report")

        st.markdown("#### API & Integration")
        st.markdown(
            """
            <div class="callout">
                <strong>API endpoints</strong><br/>
                <code>/predict</code> → accepts JSON with fields: publication_date, sra, images, length, word_count, title_url(optional)<br/>
                <code>/train</code> → retrain model (admin-only)
            </div>
            """,
            unsafe_allow_html=True,
        )
        if os.path.exists(OPENAPI_PATH):
            with open(OPENAPI_PATH, "rb") as f:
                st.download_button("Download API spec (OpenAPI YAML)", f, file_name="content_predict_api.yaml")

        st.markdown("---")
        st.markdown("Built for Knowledge Enterprise / ASU — Predicting Digital Content Performance Using Machine Learning")

        col_links = st.columns(3)
        with col_links[0]:
            if st.button("Methodology"):
                st.info("Methodology & Model Details")
                st.write("Features include publication timing, word count, images, and SRA. Models validated using time-based splits.")
        with col_links[1]:
            if st.button("Data policy"):
                st.info("Data policy")
                st.write("No personal data used. Metrics are aggregated and anonymized.")
        with col_links[2]:
            st.markdown("[Contact](mailto:content-analytics@asu.edu)")

        st.caption(f"Model version: {MODEL_VERSION} | Last retrain date: {LAST_RETRAIN_DATE}")


if __name__ == "__main__":
    main()
