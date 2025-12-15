"""
Model utilities: classical (scikit-learn) and deep learning (Keras/TensorFlow).

Public API
----------
- get_classical_models(task_type)
- train_classical_model(model_name, params, X_train, y_train)
- build_mlp(input_shape, config)
- build_cnn(input_shape, config)
- train_deep_model(model, X_train, y_train, X_val, y_val, config)
- save_model(model, path)
- load_model(path)

Notes
-----
* `task_type` ∈ {"classification", "regression"}
* Deep learning uses TensorFlow Keras if available. If TensorFlow is not
  installed, relevant functions will raise a clear ImportError.
* `save_model` and `load_model` support both sklearn (.pkl) and Keras (.h5).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import joblib
import numpy as np

# Classical ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator

# Optional TensorFlow (only needed for DL)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models, optimizers, callbacks
except Exception:
    tf = None  # graceful degradation


# -----------------------------------------------------------------------------
# Classical models registry (with docs & per-parameter help)
# -----------------------------------------------------------------------------

_CLASSICAL_MODELS: Dict[str, Dict[str, Any]] = {
    "regression": {
        "LinearRegression": {
            "cls": LinearRegression,
            "doc": "Régression linéaire ordinaire. Rapide et interprétable; baseline utile.",
            "params": {"fit_intercept": True},
            "ui": {
                "fit_intercept": {
                    "type": "checkbox",
                    "value": True,
                    "help": "Ajoute un biais (ordonnée à l’origine). Souvent recommandé."
                },
            },
        },
        "Ridge": {
            "cls": Ridge,
            "doc": "Ridge (L2) réduit la variance et stabilise les coefficients.",
            "params": {"alpha": 1.0},
            "ui": {
                "alpha": {
                    "type": "slider", "min": 0.0, "max": 10.0, "step": 0.1, "value": 1.0,
                    "help": "Force de régularisation L2 (↑ = coefficients plus petits, moins d’overfit)."
                },
            },
        },
        "Lasso": {
            "cls": Lasso,
            "doc": "Lasso (L1) réalise une sélection de variables (met certains coefficients à 0).",
            "params": {"alpha": 0.001},
            "ui": {
                "alpha": {
                    "type": "slider", "min": 0.0, "max": 1.0, "step": 0.001, "value": 0.001,
                    "help": "Force de régularisation L1 (↑ = plus de sparsité, moins de variables actives)."
                },
            },
        },
        "DecisionTreeRegressor": {
            "cls": DecisionTreeRegressor,
            "doc": "Arbre de décision pour régression. Interprétable; peut surapprendre si trop profond.",
            "params": {"max_depth": None, "random_state": 42},
            "ui": {
                "max_depth": {
                    "type": "int", "min": 1, "max": 50, "value": 5,
                    "help": "Profondeur maximale de l’arbre (limite la complexité)."
                },
            },
        },
        "RandomForestRegressor": {
            "cls": RandomForestRegressor,
            "doc": "Forêt aléatoire (bagging d’arbres). Robuste et performant par défaut.",
            "params": {"n_estimators": 200, "max_depth": None, "random_state": 42},
            "ui": {
                "n_estimators": {
                    "type": "int", "min": 10, "max": 1000, "value": 200, "step": 10,
                    "help": "Nombre d’arbres dans la forêt (↑ = plus stable, plus lent)."
                },
                "max_depth": {
                    "type": "int", "min": 1, "max": 50, "value": 10,
                    "help": "Profondeur max des arbres (contrôle l’overfit)."
                },
            },
        },
        "SVR": {
            "cls": SVR,
            "doc": "SVM pour régression. Capture des non-linéarités via le noyau choisi.",
            "params": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
            "ui": {
                "C": {
                    "type": "slider", "min": 0.1, "max": 10.0, "step": 0.1, "value": 1.0,
                    "help": "Pénalisation des erreurs (↑ = moins de régularisation)."
                },
                "epsilon": {
                    "type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.1,
                    "help": "Largeur de la zone d’insensibilité autour de la prédiction."
                },
                "kernel": {
                    "type": "select", "options": ["linear", "rbf", "poly", "sigmoid"], "value": "rbf",
                    "help": "Type de noyau (linéaire ou non linéaire)."
                },
            },
        },
        "KNNRegressor": {
            "cls": KNeighborsRegressor,
            "doc": "k-Plus Proches Voisins (régression). Non paramétrique, sensible à l’échelle des features.",
            "params": {"n_neighbors": 5},
            "ui": {
                "n_neighbors": {
                    "type": "int", "min": 1, "max": 50, "value": 5,
                    "help": "Nombre de voisins utilisés pour la prédiction."
                },
            },
        },
    },
    "classification": {
        "LogisticRegression": {
            "cls": LogisticRegression,
            "doc": "Classifieur linéaire de base, rapide et solide.",
            "params": {"C": 1.0, "max_iter": 1000},
            "ui": {
                "C": {
                    "type": "slider", "min": 0.01, "max": 10.0, "step": 0.01, "value": 1.0,
                    "help": "Inverse de la régularisation (↑ = moins de régularisation)."
                },
                "max_iter": {
                    "type": "int", "min": 100, "max": 5000, "value": 1000,
                    "help": "Nombre maximal d’itérations pour converger."
                },
            },
        },
        "DecisionTreeClassifier": {
            "cls": DecisionTreeClassifier,
            "doc": "Arbre de décision. Interprétable; prone à l’overfit si non contraint.",
            "params": {"max_depth": None, "random_state": 42},
            "ui": {
                "max_depth": {
                    "type": "int", "min": 1, "max": 50, "value": 5,
                    "help": "Profondeur maximale de l’arbre (limite la complexité)."
                },
            },
        },
        "RandomForestClassifier": {
            "cls": RandomForestClassifier,
            "doc": "Forêt aléatoire. Bon compromis biais/variance, robuste au bruit.",
            "params": {"n_estimators": 200, "max_depth": None, "random_state": 42},
            "ui": {
                "n_estimators": {
                    "type": "int", "min": 10, "max": 1000, "value": 200, "step": 10,
                    "help": "Nombre d’arbres (↑ = plus stable, plus lent)."
                },
                "max_depth": {
                    "type": "int", "min": 1, "max": 50, "value": 10,
                    "help": "Profondeur max des arbres (contrôle l’overfit)."
                },
            },
        },
        "SVM": {
            "cls": SVC,
            "doc": "SVM pour classification. Efficace avec marge maximale et noyaux non linéaires.",
            "params": {"C": 1.0, "kernel": "rbf", "probability": True},
            "ui": {
                "C": {
                    "type": "slider", "min": 0.01, "max": 10.0, "step": 0.01, "value": 1.0,
                    "help": "Pénalisation des erreurs (↑ = moins de régularisation)."
                },
                "kernel": {
                    "type": "select", "options": ["linear", "rbf", "poly", "sigmoid"], "value": "rbf",
                    "help": "Type de noyau (linéaire ou non-linéaire)."
                },
            },
        },
        "KNNClassifier": {
            "cls": KNeighborsClassifier,
            "doc": "k-Plus Proches Voisins (classification). Simple, sans entraînement lourd.",
            "params": {"n_neighbors": 5},
            "ui": {
                "n_neighbors": {
                    "type": "int", "min": 1, "max": 50, "value": 5,
                    "help": "Nombre de voisins pour voter la classe."
                },
            },
        },
        "NaiveBayes": {
            "cls": GaussianNB,
            "doc": "Naïf Bayes Gaussien. Très rapide; fonctionne bien si les features sont peu corrélées.",
            "params": {},
            "ui": {},
        },
        "GradientBoosting": {
            "cls": GradientBoostingClassifier,
            "doc": "Boosting d’arbres (séquentiel). Très performant sur tabulaire.",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "ui": {
                "n_estimators": {
                    "type": "int", "min": 50, "max": 1000, "value": 100, "step": 10,
                    "help": "Nombre d’itérations (arbres faibles) du boosting."
                },
                "learning_rate": {
                    "type": "slider", "min": 0.01, "max": 1.0, "step": 0.01, "value": 0.1,
                    "help": "Taux d’apprentissage (↓ = plus lent mais plus robuste)."
                },
            },
        },
    },
}


def get_classical_models(task_type: str) -> Dict[str, Any]:
    """Return the registry for a given task type."""
    t = (task_type or "").strip().lower()
    if t not in _CLASSICAL_MODELS:
        raise ValueError(f"Unknown task_type '{task_type}'. Use 'classification' or 'regression'.")
    return _CLASSICAL_MODELS[t]


def _instantiate_model(model_name: str, params: Dict[str, Any]) -> BaseEstimator:
    # Resolve class
    for task_dict in _CLASSICAL_MODELS.values():
        if model_name in task_dict:
            cls = task_dict[model_name]["cls"]
            default_params = task_dict[model_name].get("params", {})
            merged = {**default_params, **(params or {})}
            return cls(**merged)
    raise ValueError(f"Model '{model_name}' not found in registry.")


def train_classical_model(
    model_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Instantiate and fit a scikit-learn estimator."""
    model = _instantiate_model(model_name, params)
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------
# Keras / Deep Learning builders
# -----------------------------------------------------------------------------

def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for deep learning features. Please install tensorflow.")


def _set_tf_memory_growth():
    if tf is None:
        return
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


@dataclass
class MLPConfig:
    task_type: str  # 'classification' | 'regression'
    num_classes: int = 1
    hidden_layers: List[int] = None
    activation: str = "relu"
    dropout: float = 0.0
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 20
    metrics: List[str] = None
    auto: bool = True


@dataclass
class CNNConfig:
    task_type: str
    num_classes: int = 10
    base_filters: int = 32
    conv_blocks: int = 2
    dense_units: int = 128
    dropout: float = 0.25
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    metrics: List[str] = None
    auto: bool = True


def build_mlp(input_shape: Tuple[int, ...], config: Dict[str, Any]):
    _require_tf()
    _set_tf_memory_growth()
    if isinstance(config, dict):
        cfg = MLPConfig(**{**MLPConfig(task_type=config.get("task_type", "classification")).__dict__, **config})
    else:
        cfg = config

    model = models.Sequential(name="mlp")
    model.add(layers.Input(shape=input_shape))

    if cfg.auto or not cfg.hidden_layers:
        width = max(32, min(256, input_shape[0] * 2))
        hidden = [width, max(16, width // 2)]
    else:
        hidden = cfg.hidden_layers

    for units in hidden:
        model.add(layers.Dense(units, activation=cfg.activation))
        if cfg.dropout and cfg.dropout > 0:
            model.add(layers.Dropout(cfg.dropout))

    if cfg.task_type == "classification":
        if cfg.num_classes <= 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = cfg.metrics or ["accuracy"]
        else:
            model.add(layers.Dense(cfg.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = cfg.metrics or ["accuracy"]
    else:
        model.add(layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = cfg.metrics or ["mae"]

    opt = _build_optimizer(cfg.optimizer, cfg.learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def build_cnn(input_shape: Tuple[int, ...], config: Dict[str, Any]):
    _require_tf()
    _set_tf_memory_growth()
    if isinstance(config, dict):
        cfg = CNNConfig(**{**CNNConfig(task_type=config.get("task_type", "classification")).__dict__, **config})
    else:
        cfg = config

    model = models.Sequential(name="cnn")
    model.add(layers.Input(shape=input_shape))

    filters = cfg.base_filters
    for _ in range(cfg.conv_blocks):
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        if cfg.dropout and cfg.dropout > 0:
            model.add(layers.Dropout(cfg.dropout))
        filters = filters * 2

    model.add(layers.Flatten())
    model.add(layers.Dense(cfg.dense_units, activation="relu"))
    if cfg.dropout and cfg.dropout > 0:
        model.add(layers.Dropout(cfg.dropout))

    if cfg.task_type == "classification":
        if cfg.num_classes <= 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = cfg.metrics or ["accuracy"]
        else:
            model.add(layers.Dense(cfg.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = cfg.metrics or ["accuracy"]
    else:
        model.add(layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = cfg.metrics or ["mae"]

    opt = _build_optimizer(cfg.optimizer, cfg.learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def _build_optimizer(name: str, lr: float):
    if tf is None:
        raise ImportError("TensorFlow required for deep learning optimizers.")
    name = (name or "adam").lower()
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    return optimizers.Adam(learning_rate=lr)


def train_deep_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    config: Dict[str, Any],
):
    _require_tf()
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    validation_split = float(config.get("validation_split", 0.0))

    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    user_cbs = config.get("callbacks", [])
    if user_cbs:
        cbs.extend(user_cbs)

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=cbs,
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=cbs,
        )

    return model, history


def save_model(model: Any, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_keras = hasattr(model, "save") and ("keras" in str(type(model)).lower())
    if p.suffix == ".pkl":
        joblib.dump(model, p)
        return str(p)
    elif p.suffix == ".h5":
        if not is_keras:
            raise ValueError(".h5 est réservé aux modèles Keras.")
        model.save(str(p))
        return str(p)
    if is_keras:
        p = p.with_suffix(".h5")
        model.save(str(p))
    else:
        p = p.with_suffix(".pkl")
        joblib.dump(model, p)
    return str(p)


def load_model(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if p.suffix == ".pkl":
        return joblib.load(p)
    elif p.suffix == ".h5":
        if tf is None:
            raise ImportError("TensorFlow required to load .h5 Keras models.")
        # 👉 Évite la désérialisation des métriques/optimiseur non compatibles
        try:
            return tf.keras.models.load_model(str(p), compile=False)
        except Exception:
            # Dernier recours: quelques alias fréquents (facultatif)
            custom = {
                "mse": tf.keras.losses.MeanSquaredError(),
                "mae": tf.keras.losses.MeanAbsoluteError(),
                "accuracy": tf.keras.metrics.Accuracy(),
                "acc": tf.keras.metrics.Accuracy(),
            }
            return tf.keras.models.load_model(str(p), custom_objects=custom, compile=False)
    else:
        raise ValueError("Unknown model extension. Use .pkl (sklearn) or .h5 (Keras).")

