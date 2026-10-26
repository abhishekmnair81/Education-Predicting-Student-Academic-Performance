import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]

REQUIRED_COLUMNS = CATEGORICAL_COLS + [c for c in NUMERIC_COLS if c != "G3"] + ["G3"]

@st.cache_data(show_spinner=False)
def load_dataset(file_source=None):
    """
    Load dataset from a file path or uploaded file buffer.
    Detects semicolon or comma delimiters.
    Returns cleaned DataFrame with Target and Final_Status columns.
    """
    if file_source is None:
        file_path = "student-mat.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Default dataset '{file_path}' was not found in directory.")
        df = pd.read_csv(file_path, sep=";")
    elif isinstance(file_source, (str, os.PathLike)):
        df = pd.read_csv(file_source, sep=";")
    else:
        # Uploaded file buffer (BytesIO or StringIO)
        content = file_source.getvalue()
        try:
            # Try semicolon first (UCI standard)
            df = pd.read_csv(io.BytesIO(content), sep=";")
            if df.shape[1] <= 1:
                df = pd.read_csv(io.BytesIO(content), sep=",")
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=",")

    # Clean string column stripping
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.replace('"', '')

    # Convert numeric columns safely
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any numeric anomalies with column median if necessary (standard dataset has 0 missing)
    if "G3" in df.columns:
        df["Target"] = np.where(df["G3"] >= 10, 1, 0)
        df["Final_Status"] = np.where(df["G3"] >= 10, "Pass", "Fail")
    else:
        raise ValueError("Dataset must contain the 'G3' final grade column to define Pass/Fail target.")

    return df

def get_data_summary(df):
    """
    Generate dataset summary statistics, shape, missing values, duplicates.
    """
    total_rows, total_cols = df.shape
    missing_counts = int(df.isnull().sum().sum())
    dup_counts = int(df.duplicated().sum())
    
    pass_count = int((df["Target"] == 1).sum()) if "Target" in df.columns else 0
    fail_count = int((df["Target"] == 0).sum()) if "Target" in df.columns else 0
    pass_rate = (pass_count / total_rows * 100) if total_rows > 0 else 0.0

    dtypes_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique().values,
        "Sample Value": [df[c].iloc[0] if len(df) > 0 else None for c in df.columns]
    })

    return {
        "rows": total_rows,
        "cols": total_cols,
        "missing": missing_counts,
        "duplicates": dup_counts,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "dtypes_summary": dtypes_summary
    }

def prepare_features(df, include_g1_g2=True):
    """
    Extracts predictor feature matrix X and target y.
    Prevents data leakage by dropping G3, Final_Status, Target.
    Optionally drops G1 and G2 based on user toggle.
    Encodes 17 categorical columns with LabelEncoder.
    Returns:
        X_encoded (DataFrame)
        y (Series)
        encoders_dict (dict)
        feature_names (list)
        X_raw_preview (DataFrame)
    """
    features_to_drop = ["G3", "Final_Status", "Target"]
    if not include_g1_g2:
        features_to_drop.extend(["G1", "G2"])

    drop_existing = [c for c in features_to_drop if c in df.columns]
    X_raw = df.drop(columns=drop_existing).copy()
    y = df["Target"].copy()

    encoders_dict = {}
    X_encoded = X_raw.copy()

    for col in CATEGORICAL_COLS:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_raw[col].astype(str))
            encoders_dict[col] = le

    # Ensure all remaining columns are float/int
    for col in X_encoded.columns:
        X_encoded[col] = pd.to_numeric(X_encoded[col])

    return X_encoded, y, encoders_dict, list(X_encoded.columns), X_raw

def filter_dataframe(df, filters):
    """
    Applies sidebar filters reactively to the dataframe.
    filters dict expected keys:
        school, sex, age_range, address, failures, studytime,
        higher, internet, romantic, absences_range
    """
    filtered = df.copy()

    if filters.get("school") and filters["school"] != "All":
        filtered = filtered[filtered["school"] == filters["school"]]

    if filters.get("sex") and filters["sex"] != "All":
        filtered = filtered[filtered["sex"] == filters["sex"]]

    if filters.get("age_range"):
        min_age, max_age = filters["age_range"]
        filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]

    if filters.get("address") and filters["address"] != "All":
        filtered = filtered[filtered["address"] == filters["address"]]

    if filters.get("failures") and filters["failures"] != "All":
        val = int(filters["failures"])
        filtered = filtered[filtered["failures"] == val]

    if filters.get("studytime") and filters["studytime"] != "All":
        val = int(filters["studytime"])
        filtered = filtered[filtered["studytime"] == val]

    if filters.get("higher") and filters["higher"] != "All":
        filtered = filtered[filtered["higher"] == filters["higher"]]

    if filters.get("internet") and filters["internet"] != "All":
        filtered = filtered[filtered["internet"] == filters["internet"]]

    if filters.get("romantic") and filters["romantic"] != "All":
        filtered = filtered[filtered["romantic"] == filters["romantic"]]

    if filters.get("absences_range"):
        min_abs, max_abs = filters["absences_range"]
        filtered = filtered[(filtered["absences"] >= min_abs) & (filtered["absences"] <= max_abs)]

    return filtered
