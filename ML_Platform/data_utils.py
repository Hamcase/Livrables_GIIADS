"""
Data utilities: loading, cleaning, splitting and preprocessing for
both tabular and (optionally) image datasets.

Public API (as requested):
- load_csv(uploaded_file)
- load_builtin_dataset(name)
- clean_data(df)
- split_data(X, y, test_size, stratify)
- preprocess_data(X, y)

Notes
-----
* `load_builtin_dataset` supports common scikit-learn tabular datasets
  (iris, wine, breast_cancer, diabetes, california_housing) and two
  image datasets via TensorFlow (mnist, cifar10). It returns a tuple
  `(payload, meta)` where:
    - for tabular: `payload` is a pandas DataFrame; `meta = {"modality": "tabular", "target_name": <str or None>}`
    - for image:  `payload` is a dict with numpy arrays (X_train, y_train, X_test, y_test);
                  `meta = {"modality": "image", "num_classes": int, "input_shape": tuple}`
* `clean_data` performs basic sanitation for tabular data.
* `preprocess_data` builds a ColumnTransformer pipeline (impute + encode + scale)
  and returns processed features/labels along with artifacts to reuse for inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import io
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Optional TF import only when needed for image datasets
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # Graceful degradation if TF is not installed


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded via Streamlit.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The file object returned by `st.file_uploader`.

    Returns
    -------
    pd.DataFrame
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")

    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    elif name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")


TABULAR_DATASETS = {
    "iris": "sklearn.datasets.load_iris",
    "wine": "sklearn.datasets.load_wine",
    "breast_cancer": "sklearn.datasets.load_breast_cancer",
    "diabetes": "sklearn.datasets.load_diabetes",
    "california_housing": "sklearn.datasets.fetch_california_housing",
}

IMAGE_DATASETS = {
    "mnist": "tf.keras.datasets.mnist",
    "cifar10": "tf.keras.datasets.cifar10",
}


def _import_sklearn_loader(path: str):
    from importlib import import_module

    module_path, func_name = path.rsplit(".", 1)
    mod = import_module(module_path)
    return getattr(mod, func_name)


def load_builtin_dataset(name: str):
    """Load a built-in dataset by name.

    For tabular datasets, returns `(df, meta)`.
    For image datasets, returns `({"X_train", "y_train", "X_test", "y_test"}, meta)`.

    The `meta` dict always includes a `modality` key among optional others.
    """
    key = (name or "").strip().lower()

    # --- Tabular datasets via scikit-learn ---
    if key in TABULAR_DATASETS:
        loader = _import_sklearn_loader(TABULAR_DATASETS[key])
        bunch = loader()
        if hasattr(bunch, "frame") and bunch.frame is not None:
            df = bunch.frame.copy()
        else:
            # Build DataFrame manually
            X = bunch.data
            feature_names = getattr(bunch, "feature_names", [f"f{i}" for i in range(X.shape[1])])
            df = pd.DataFrame(X, columns=feature_names)
            target = pd.Series(bunch.target, name="target")
            df[target.name] = target
        target_name = getattr(bunch, "target_names", None)
        return df, {"modality": "tabular", "target_name": "target"}

    # --- Image datasets via TensorFlow ---
    if key in IMAGE_DATASETS:
        if tf is None:
            raise ImportError(
                "TensorFlow is not available. Install tensorflow to use image datasets."
            )
        ds = IMAGE_DATASETS[key]
        if ds.endswith("mnist"):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            # Expand channel dim → (H, W, 1)
            X_train = X_train[..., np.newaxis].astype("float32") / 255.0
            X_test = X_test[..., np.newaxis].astype("float32") / 255.0
            input_shape = X_train.shape[1:]
            num_classes = int(np.max(y_train)) + 1
        elif ds.endswith("cifar10"):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0
            input_shape = X_train.shape[1:]
            num_classes = int(np.max(y_train)) + 1
        else:
            raise ValueError(f"Unknown image dataset: {name}")

        payload = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        meta = {"modality": "image", "num_classes": num_classes, "input_shape": input_shape}
        return payload, meta

    raise ValueError(
        f"Dataset '{name}' non reconnu. Options tabulaires: {list(TABULAR_DATASETS)}; "
        f"images: {list(IMAGE_DATASETS)}"
    )


# -----------------------------------------------------------------------------
# Cleaning for tabular data
# -----------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanitation: drop all-NaN columns, strip column names,
    try to coerce numeric columns, and standardize booleans.
    """
    df = df.copy()

    # Trim column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully-empty columns
    df = df.dropna(axis=1, how="all")

    # Coerce obvious numeric strings to numbers when safe
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric coercion; keep as object if too many NaNs introduced
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Heuristic: accept if at least 50% non-null after coercion
            if coerced.notna().mean() >= 0.5:
                df[col] = coerced

    # Normalize boolean-like strings
    TRUE_SET = {"true", "yes", "y", "vrai", "1"}
    FALSE_SET = {"false", "no", "n", "faux", "0"}
    for col in df.columns:
        if df[col].dtype == object:
            lower = df[col].astype(str).str.lower().str.strip()
            if lower.isin(TRUE_SET | FALSE_SET).mean() > 0.8:
                df[col] = lower.isin(TRUE_SET).astype(int)

    return df


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split with optional stratification.

    If `stratify=True`, uses `y` for stratification (better for classification).
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# Preprocessing for tabular data
# -----------------------------------------------------------------------------

def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Fit/transform preprocessing for tabular data.

    Steps:
    - Numerical: impute median → scale (StandardScaler)
    - Categorical: impute most_frequent → OneHot
    - y handling: if y is non-numeric/categories → LabelEncoder

    Returns
    -------
    X_processed : np.ndarray
    y_processed : Optional[np.ndarray]
    artifacts : dict
        Contains the fitted `preprocessor` (ColumnTransformer) and
        optional `label_encoder`, plus feature metadata.
    """
    X = X.copy()

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Define pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    y_processed = None
    label_encoder = None
    if y is not None:
        if not np.issubdtype(np.asarray(y).dtype, np.number):
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y.astype(str))
        else:
            y_processed = y.to_numpy()

    artifacts = {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_out_dim": X_processed.shape[1],
    }

    return X_processed, y_processed, artifacts
