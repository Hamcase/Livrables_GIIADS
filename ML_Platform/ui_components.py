"""
Streamlit UI components used by main_app.py (avec aide/explications des paramètres,
chargement / upload de modèles, et recompilation Keras si nécessaire)

Public API
----------
- upload_data_ui()
- select_task_ui()
- model_selection_ui()
- training_ui()
- evaluation_ui()
- prediction_ui()

Chaque fonction lit/écrit depuis `st.session_state` pour garder un flux clair.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import (
    load_csv,
    load_builtin_dataset,
    clean_data,
    split_data,
    preprocess_data,
)
from model_utils import (
    get_classical_models,
    train_classical_model,
    build_mlp,
    build_cnn,
    train_deep_model,
    load_model,  # <- pour charger .pkl / .h5 existants
)
from evaluation_utils import (
    evaluate_model,
    plot_confusion_matrix,
    plot_regression_results,
    plot_training_curves,
    get_metrics_report,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_image_modality(meta: Dict[str, Any]) -> bool:
    return meta and meta.get("modality") == "image"


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Aucun DataFrame disponible.")
    if df.empty:
        raise ValueError("Le DataFrame est vide.")
    return df


def _list_saved_models() -> List[str]:
    """Liste les modèles présents dans ./models avec extensions supportées."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    files: List[str] = []
    for ext in ("*.pkl", "*.h5"):
        files.extend([str(p) for p in models_dir.glob(ext)])
    return sorted(files)


def _is_keras_model(m: Any) -> bool:
    # Heuristique simple pour distinguer un modèle Keras
    return hasattr(m, "predict") and hasattr(m, "save") and hasattr(m, "get_config")


def _maybe_compile_keras(model: Any, task: str, num_classes: Optional[int] = None, lr: float = 1e-3):
    """Compile le modèle Keras si nécessaire (après load_model(..., compile=False))."""
    try:
        already_compiled = getattr(model, "compiled_loss", None) is not None
    except Exception:
        already_compiled = False
    if already_compiled:
        return model

    import tensorflow as tf  # import local (optionnel si pas de deep)

    if task == "classification":
        if num_classes is None or num_classes <= 2:
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = [tf.keras.metrics.BinaryAccuracy()]
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    else:
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


# -----------------------------------------------------------------------------
# 1) Upload / Jeux intégrés
# -----------------------------------------------------------------------------

def upload_data_ui():
    """Handle data loading (CSV/Excel) or built-in datasets.

    Returns
    -------
    tuple[pd.DataFrame, str] | None
        (df, dataset_name) when new data is loaded, else None.
    """
    st.subheader("Source des données")
    tab1, tab2 = st.tabs(["📤 Upload", "📚 Jeux intégrés"])

    # --- Upload ---
    with tab1:
        up = st.file_uploader("Uploader un fichier CSV/TSV/Excel",
                              type=["csv", "tsv", "txt", "xlsx", "xls"])
        if up is not None:
            try:
                df = load_csv(up)
                df = clean_data(df)
                st.write("**Aperçu** :", df.shape)
                st.dataframe(df.head())
                return df, up.name
            except Exception as e:
                st.error(f"Échec du chargement: {e}")

    # --- Built-ins (MNIST retiré) ---
    with tab2:
        builtin_choice = st.selectbox(
            "Choisir un dataset intégré",
            ["None", "iris", "wine", "breast_cancer", "diabetes", "california_housing", "cifar10"],
            index=0,
        )
        if builtin_choice != "None":
            try:
                payload, meta = load_builtin_dataset(builtin_choice)
                if _is_image_modality(meta):
                    # Image dataset : on stocke dans le state (pas de DataFrame)
                    st.session_state.df = None
                    st.session_state.dataset_name = builtin_choice
                    st.session_state.image_data = payload
                    st.session_state.image_meta = meta
                    st.success(
                        f"Dataset image chargé: {builtin_choice} → input_shape={meta['input_shape']}, "
                        f"classes={meta['num_classes']}\n"
                        "Passez directement aux étapes 3–4 pour choisir CNN et entraîner."
                    )
                    return None
                else:
                    df, _ = payload, meta
                    df = clean_data(df)
                    st.write("**Aperçu** :", df.shape)
                    st.dataframe(df.head())
                    return df, builtin_choice
            except Exception as e:
                st.error(f"Échec du chargement intégré: {e}")

    # Si déjà chargé avant, on récap
    if st.session_state.get("df") is not None:
        df = st.session_state.df
        st.write("**Aperçu** :", df.shape)
        st.dataframe(df.head())

    return None


# -----------------------------------------------------------------------------
# 2) Tâche & Prétraitements (tabulaire uniquement)
# -----------------------------------------------------------------------------

def select_task_ui():
    df = _ensure_df(st.session_state.df)

    st.subheader("Type de tâche & colonnes")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.task_type = st.selectbox("Tâche", ["classification", "regression"], index=0)
    with c2:
        st.session_state.approach = st.selectbox("Approche", ["classical", "deep"], index=0)

    # Sélection colonnes
    all_cols = df.columns.tolist()
    with st.expander("Sélection manuelle des colonnes", expanded=True):
        target = st.selectbox("Colonne cible", all_cols, index=len(all_cols)-1)
        feats = st.multiselect("Colonnes features", [c for c in all_cols if c != target],
                               default=[c for c in all_cols if c != target])
        st.session_state.target_col = target
        st.session_state.feature_cols = feats

    st.markdown("---")
    st.write("**Shape**:", df.shape)
    st.write("**Types**:")
    st.dataframe(pd.DataFrame({"col": df.columns, "dtype": [str(t) for t in df.dtypes]}))

    # Split + preprocessing
    st.subheader("Split train/test")
    c3, c4, c5 = st.columns([2, 1, 1])
    with c3:
        test_size = st.slider("test_size", 0.1, 0.5, float(st.session_state.get("test_size", 0.2)), 0.05,
                              help="Proportion des données réservée au test.")
    with c4:
        stratify = st.checkbox("Stratify (classification)",
                               value=bool(st.session_state.get("stratify", False)),
                               help="Assure une même répartition des classes entre train/test.")
    with c5:
        if st.button("Appliquer split & preprocessing", use_container_width=True):
            try:
                X = df[st.session_state.feature_cols].copy()
                y = df[st.session_state.target_col].copy()

                X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(
                    X, y, test_size=test_size,
                    stratify=stratify and st.session_state.task_type == "classification"
                )
                X_train_proc, y_train_proc, artifacts = preprocess_data(X_train_raw, y_train_raw)
                X_test_proc = artifacts["preprocessor"].transform(X_test_raw)

                y_test_proc = y_test_raw
                if artifacts.get("label_encoder") is not None:
                    y_test_proc = artifacts["label_encoder"].transform(y_test_raw.astype(str))

                # Save
                st.session_state.test_size = test_size
                st.session_state.stratify = stratify
                st.session_state.X_train = X_train_proc
                st.session_state.X_test = X_test_proc
                st.session_state.y_train = np.asarray(y_train_proc)
                st.session_state.y_test = np.asarray(y_test_proc)
                st.session_state.preprocess_artifacts = artifacts

                st.success("Split & preprocessing appliqués ✔")
            except Exception as e:
                st.error(f"Erreur split/preprocessing : {e}")


# -----------------------------------------------------------------------------
# 3) Sélection du modèle (classical ou deep) + chargement / upload
# -----------------------------------------------------------------------------

def model_selection_ui():
    st.subheader("Sélection du modèle")
    approach = st.session_state.get("approach")
    task = st.session_state.get("task_type")

    # --- Charger un modèle depuis ./models OU uploader un modèle ---
    with st.expander("📁 Charger un modèle existant (./models)", expanded=False):
        saved = _list_saved_models()
        if not saved:
            st.caption("Aucun fichier trouvé dans ./models (attendus: .pkl pour sklearn, .h5 pour Keras)")
        else:
            choice = st.selectbox("Fichier modèle", saved, key="saved_model_choice")
            colA, colB = st.columns([1, 1])
            with colA:
                load_btn = st.button("Charger ce modèle")
            with colB:
                st.caption("Astuce: pour les modèles tabulaires, refaites l'étape 2 avec **les mêmes features** pour disposer du préprocesseur.")
            if load_btn and choice:
                try:
                    mdl = load_model(choice)  # compile=False pour .h5 (géré dans model_utils)
                    st.session_state.model = mdl
                    st.session_state.model_name = f"(loaded) {Path(choice).name}"
                    if choice.endswith(".h5"):
                        st.session_state.approach = "deep"
                        st.info("Modèle Keras chargé **sans compilation** (compatibilité). "
                                "Il sera re-compilé automatiquement si vous relancez un entraînement.")
                    else:
                        st.session_state.approach = "classical"
                    st.session_state.model_params = {}
                    st.success(f"Modèle chargé depuis {choice}")
                    return  # on sort après chargement
                except Exception as e:
                    st.error(f"Échec du chargement: {e}")

        st.markdown("---")
        st.caption("Ou **uploadez** un fichier modèle (.pkl / .h5) :")
        up_model = st.file_uploader("Uploader un modèle", type=["pkl", "h5"], key="upload_model_file")
        colU1, colU2 = st.columns([1, 2])
        with colU1:
            import_btn = st.button("Importer & charger")
        with colU2:
            st.caption("Le fichier sera copié dans ./models puis chargé. "
                       "Pour sklearn, pensez au préprocesseur (Étape 2).")
        if import_btn and up_model is not None:
            try:
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)
                safe_name = up_model.name.replace("..", "_").replace("/", "_").replace("\\", "_")
                dest = models_dir / safe_name
                with open(dest, "wb") as f:
                    f.write(up_model.read())
                mdl = load_model(str(dest))
                st.session_state.model = mdl
                st.session_state.model_name = f"(uploaded) {safe_name}"
                if str(dest).endswith(".h5"):
                    st.session_state.approach = "deep"
                    st.info("Modèle Keras chargé **sans compilation** (compatibilité). "
                            "Il sera re-compilé automatiquement si vous relancez un entraînement.")
                else:
                    st.session_state.approach = "classical"
                st.session_state.model_params = {}
                st.success(f"Modèle importé et chargé: {dest}")
                return
            except Exception as e:
                st.error(f"Échec de l'import/chargement: {e}")

    # --- Sélection et paramétrage standard ---
    if approach == "classical":
        registry = get_classical_models(task)
        model_name = st.selectbox("Modèle", list(registry.keys()))
        st.session_state.model_name = model_name

        ui_spec = registry[model_name].get("ui", {})

        # Panneau d'explication
        with st.expander("ℹ️ À propos de ce modèle", expanded=False):
            desc = registry[model_name].get("doc")
            if desc:
                st.write(desc)
            if ui_spec:
                st.markdown("**Paramètres disponibles** :")
                for pname, spec in ui_spec.items():
                    help_txt = spec.get("help", "")
                    st.caption(f"`{pname}` — {help_txt}")

        # Widgets + tooltips
        params: Dict[str, Any] = {}
        with st.form("params_form"):
            for pname, spec in ui_spec.items():
                ptype = spec.get("type")
                help_txt = spec.get("help")
                if ptype == "slider":
                    params[pname] = st.slider(
                        pname,
                        min_value=float(spec.get("min", 0.0)),
                        max_value=float(spec.get("max", 1.0)),
                        value=float(spec.get("value", 0.5)),
                        step=float(spec.get("step", 0.01)),
                        help=help_txt,
                    )
                elif ptype == "int":
                    params[pname] = st.number_input(
                        pname,
                        min_value=int(spec.get("min", 1)),
                        max_value=int(spec.get("max", 1000)),
                        value=int(spec.get("value", 10)),
                        step=int(spec.get("step", 1)),
                        help=help_txt,
                    )
                elif ptype == "checkbox":
                    params[pname] = st.checkbox(pname, value=bool(spec.get("value", False)), help=help_txt)
                elif ptype == "select":
                    options = spec.get("options", [])
                    default_idx = options.index(spec.get("value")) if spec.get("value") in options else 0
                    params[pname] = st.selectbox(pname, options=options, index=default_idx, help=help_txt)
                else:
                    st.write(f"Paramètre non géré: {pname}")

            submitted = st.form_submit_button("Valider hyperparamètres")
        if submitted:
            st.session_state.model_params = params
            st.success("Hyperparamètres enregistrés ✔")

    else:  # deep
        st.session_state.model_name = "MLP" if st.session_state.get("df") is not None else "CNN"
        deep_type = st.selectbox("Type de réseau", ["MLP (tabulaire)", "CNN (images)"])
        if deep_type.startswith("MLP"):
            cfg = {"task_type": task}
            cfg["auto"] = st.checkbox("Auto-build (recommandé)", value=True,
                                      help="Génère automatiquement une architecture adaptée à la taille d'entrée.")
            if not cfg["auto"]:
                layers_txt = st.text_input("Hidden layers (ex: 128,64)", value="128,64",
                                           help="Tailles des couches denses successives.")
                try:
                    cfg["hidden_layers"] = [int(x.strip()) for x in layers_txt.split(",") if x.strip()]
                except Exception:
                    st.warning("Format invalide, fallback auto-build.")
                    cfg["auto"] = True
            colA, colB, colC = st.columns(3)
            with colA:
                cfg["activation"] = st.selectbox("Activation", ["relu", "tanh", "gelu"], index=0,
                                                 help="Fonction d'activation des couches cachées.")
            with colB:
                cfg["dropout"] = st.slider("Dropout", 0.0, 0.8, 0.0, 0.05,
                                           help="Régularisation: fraction de neurones éteints pendant l'entraînement.")
            with colC:
                cfg["learning_rate"] = st.slider("Learning rate", 1e-4, 1e-1, 1e-3, format="%.5f",
                                                 help="Pas d'apprentissage de l'optimiseur.")
            cfg["batch_size"] = st.number_input("Batch size", 8, 512, 32, 8,
                                                help="Nombre d'exemples par mise à jour de gradient.")
            cfg["epochs"] = st.number_input("Epochs", 1, 200, 20, 1,
                                            help="Nombre de passes complètes sur le dataset.")
            st.session_state.model_params = cfg
            st.session_state.model_name = "MLP"
        else:
            cfg = {"task_type": "classification", "auto": True}
            col1, col2, col3 = st.columns(3)
            with col1:
                cfg["base_filters"] = st.number_input("Base filters", 8, 128, 32, 8,
                                                      help="Nombre de filtres du premier bloc conv (double à chaque bloc).")
            with col2:
                cfg["conv_blocks"] = st.number_input("Conv blocks", 1, 4, 2, 1,
                                                     help="Nombre de blocs Conv→Conv→Pool.")
            with col3:
                cfg["dense_units"] = st.number_input("Dense units", 16, 512, 128, 16,
                                                     help="Taille de la couche dense finale avant la sortie.")
            cfg["dropout"] = st.slider("Dropout", 0.0, 0.8, 0.25, 0.05, help="Régularisation par Dropout dans le CNN.")
            cfg["learning_rate"] = st.slider("Learning rate", 1e-4, 1e-1, 1e-3, format="%.5f",
                                             help="Pas d'apprentissage de l'optimiseur.")
            cfg["batch_size"] = st.number_input("Batch size", 8, 512, 32, 8, help="Nombre d'images par batch.")
            cfg["epochs"] = st.number_input("Epochs", 1, 200, 10, 1, help="Nombre d'époques d'entraînement.")
            st.session_state.model_params = cfg
            st.session_state.model_name = "CNN"


# -----------------------------------------------------------------------------
# 4) Entraînement
# -----------------------------------------------------------------------------

def training_ui():
    st.subheader("Entraînement")

    approach = st.session_state.get("approach")
    task = st.session_state.get("task_type")

    if approach == "classical":
        if st.button("🚀 Entraîner", use_container_width=True):
            try:
                model = train_classical_model(
                    st.session_state.model_name,
                    st.session_state.get("model_params", {}),
                    st.session_state.X_train,
                    st.session_state.y_train,
                )
                st.session_state.model = model
                st.success("Modèle entraîné ✔")
            except Exception as e:
                st.error(f"Échec entraînement: {e}")

    else:
        try:
            if st.session_state.model_name == "MLP":
                input_dim = st.session_state.X_train.shape[1]
                cfg = st.session_state.get("model_params", {})
                if task == "classification":
                    cfg["num_classes"] = int(np.max(st.session_state.y_train)) + 1

                # Si un modèle Keras a été chargé (h5), on le recompile si besoin. Sinon on en crée un.
                model = st.session_state.get("model")
                if _is_keras_model(model):
                    _maybe_compile_keras(model, task, num_classes=cfg.get("num_classes"),
                                         lr=cfg.get("learning_rate", 1e-3))
                else:
                    model = build_mlp((input_dim,), cfg)

                model, history = train_deep_model(
                    model,
                    st.session_state.X_train,
                    st.session_state.y_train,
                    None,
                    None,
                    cfg,
                )
                st.session_state.model = model
                st.session_state.history = history
                st.success("MLP entraîné ✔")

            elif st.session_state.model_name == "CNN":
                data = st.session_state.get("image_data")
                meta = st.session_state.get("image_meta", {})
                if not data:
                    st.error("Aucune donnée image en mémoire. Allez à l'étape 1 et chargez CIFAR-10, ou implémentez l'upload image.")
                    return
                cfg = st.session_state.get("model_params", {})
                cfg["num_classes"] = meta.get("num_classes", 10)

                model = st.session_state.get("model")
                if _is_keras_model(model):
                    _maybe_compile_keras(model, "classification",
                                         num_classes=cfg["num_classes"],
                                         lr=cfg.get("learning_rate", 1e-3))
                else:
                    model = build_cnn(meta.get("input_shape"), cfg)

                model, history = train_deep_model(
                    model,
                    data["X_train"], data["y_train"],
                    data.get("X_test"), data.get("y_test"),  # val rapide sur test pour démo
                    cfg,
                )
                st.session_state.model = model
                st.session_state.history = history
                st.success("CNN entraîné ✔")

            # Courbes d'entraînement (si dispo)
            fig = plot_training_curves(st.session_state.get("history"))
            if fig is not None:
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur DL: {e}")


# -----------------------------------------------------------------------------
# 5) Évaluation
# -----------------------------------------------------------------------------

def evaluation_ui():
    st.subheader("Évaluation")

    task = st.session_state.get("task_type")

    try:
        if st.session_state.get("approach") == "classical" or st.session_state.model_name == "MLP":
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
        else:
            # CNN (images)
            data = st.session_state.get("image_data")
            X_test, y_test = data.get("X_test"), data.get("y_test")

        out = evaluate_model(st.session_state.model, X_test, y_test, task)
        st.session_state.metrics = out["metrics"]

        st.write("**Métriques**")
        st.json(out["metrics"])

        if task == "classification":
            fig = plot_confusion_matrix(y_test, out["y_pred"])
            st.pyplot(fig, use_container_width=True)
        else:
            fig1, fig2 = plot_regression_results(y_test, out["y_pred"])
            st.pyplot(fig1, use_container_width=True)
            st.pyplot(fig2, use_container_width=True)

        report = get_metrics_report(task, out["metrics"])
        st.session_state.report = report
        st.download_button(
            "⬇️ Télécharger le rapport (JSON)",
            data=json.dumps(report, indent=2),
            file_name="metrics_report.json",
            mime="application/json",
        )
    except Exception as e:
        st.error(f"Erreur d'évaluation: {e}")


# -----------------------------------------------------------------------------
# 6) Prédiction
# -----------------------------------------------------------------------------

def prediction_ui():
    st.subheader("Prédictions")

    approach = st.session_state.get("approach")
    task = st.session_state.get("task_type")

    if approach == "classical" or st.session_state.model_name == "MLP":
        # Formulaire (tabulaire)
        artifacts = st.session_state.get("preprocess_artifacts")
        if artifacts is None:
            st.info("Aucun préprocesseur connu. Assurez-vous d'avoir fait l'étape 2.")
            return
        pre = artifacts["preprocessor"]
        num_cols = artifacts["numeric_cols"]
        cat_cols = artifacts["categorical_cols"]

        with st.form("predict_form"):
            st.write("Entrer une observation")
            inputs = {}
            cols = st.columns(2)
            for i, c in enumerate(num_cols + cat_cols):
                with cols[i % 2]:
                    if c in num_cols:
                        val = st.number_input(c, value=0.0, format="%.5f")
                    else:
                        val = st.text_input(c, value="")
                    inputs[c] = val
            submitted = st.form_submit_button("Prédire")

        if submitted:
            try:
                row = pd.DataFrame([inputs])
                X_proc = pre.transform(row)
                y_pred = st.session_state.model.predict(X_proc)
                # Decode label si besoin
                if task == "classification" and artifacts.get("label_encoder") is not None and y_pred.ndim == 1:
                    inv = artifacts["label_encoder"].inverse_transform(np.asarray(y_pred).astype(int))
                    st.success(f"Prédiction: {inv[0]}")
                else:
                    st.success(f"Prédiction: {float(np.asarray(y_pred).ravel()[0]):.6f}")
            except Exception as e:
                st.error(f"Échec de prédiction: {e}")

        st.markdown("---")
        st.caption("Uploader un batch CSV pour prédire plusieurs lignes (mêmes colonnes features que l'entraînement).")
        csv_file = st.file_uploader("Batch CSV (features uniquement)", type=["csv"], key="predict_batch")
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                Xp = pre.transform(df)
                yp = st.session_state.model.predict(Xp)
                out_df = df.copy()
                out_df["prediction"] = yp if yp.ndim == 1 else yp.argmax(axis=1)
                st.dataframe(out_df.head())
                st.download_button("⬇️ Télécharger les prédictions", out_df.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prédiction échouée: {e}")

    else:
        # Prédiction image (CNN)
        st.write("Charger une image (ou une petite batch .npz) pour prédire.")
        img_file = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            try:
                from PIL import Image
                im = Image.open(img_file).convert("RGB")
                meta = st.session_state.get("image_meta", {})
                input_shape = meta.get("input_shape")
                if not input_shape:
                    st.error("input_shape inconnu. Entraînez un CNN sur CIFAR-10 pour définir la taille.")
                    return
                # Resize
                im = im.resize((input_shape[1], input_shape[0])) if len(input_shape) == 3 else im
                arr = np.array(im).astype("float32") / 255.0
                if arr.ndim == 2:
                    arr = arr[..., None]
                arr = np.expand_dims(arr, axis=0)
                proba = st.session_state.model.predict(arr, verbose=0)
                pred = int(np.argmax(proba, axis=1)[0]) if proba.ndim == 2 else int(proba.ravel()[0] >= 0.5)
                st.success(f"Classe prédite: {pred}")
            except Exception as e:
                st.error(f"Échec prédiction image: {e}")
