"""
Main entry point for the Streamlit ML/DL workbench.

This app is intentionally structured in clear sections and delegates most
logic to helper modules:
- data_utils.py          → data loading / cleaning / preprocessing
- model_utils.py         → model building / training / saving / loading
- evaluation_utils.py    → metrics & plots
- ui_components.py       → Streamlit UI building blocks

Run:  streamlit run main_app.py
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML & DL Workbench",
    page_icon="🤖",
    layout="wide",
)

# === Imports from project modules (implemented in the next steps) ===
# NOTE: These imports will work once you add the other files.
try:
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
        save_model,
        load_model,
    )
    from evaluation_utils import (
        evaluate_model,
        plot_confusion_matrix,
        plot_regression_results,
        plot_training_curves,
        get_metrics_report,
    )
    from ui_components import (
        upload_data_ui,
        select_task_ui,
        model_selection_ui,
        training_ui,
        evaluation_ui,
        prediction_ui,
    )
except Exception as e:
    # Soft warning so the file can still open before the other modules exist.
    st.sidebar.warning(
        "Les modules utilitaires seront ajoutés aux étapes suivantes.\n"
        "Certaines sections peuvent être inactives tant qu'ils ne sont pas créés."
    )


# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
DEFAULT_STATE: Dict[str, Any] = {
    # Data
    "df": None,
    "dataset_name": None,
    "target_col": None,
    "feature_cols": None,
    "task_type": None,  # "classification" | "regression"
    "approach": None,   # "classical" | "deep"

    # Splits & preprocessing
    "test_size": 0.2,
    "stratify": False,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "preprocess_artifacts": None,  # encoders/scalers to reuse at inference

    # Modeling
    "model_name": None,
    "model_params": {},
    "model": None,
    "history": None,  # Keras History

    # Evaluation
    "metrics": None,
    "report": None,

    # Prediction cache
    "pred_sample": None,
}


def init_session_state() -> None:
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_after_data_change() -> None:
    """Reset dependent artifacts when data changes."""
    for k in [
        "feature_cols",
        "target_col",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "preprocess_artifacts",
        "model_name",
        "model_params",
        "model",
        "history",
        "metrics",
        "report",
        "pred_sample",
    ]:
        st.session_state[k] = DEFAULT_STATE[k]


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------
SIDEBAR_STEPS = [
    "1. Chargement des données",
    "2. Préparation des données",
    "3. Sélection du modèle",
    "4. Entraînement",
    "5. Évaluation",
    "6. Prédiction",
    "7. Export & Réutilisation",
]


def sidebar_nav() -> str:
    st.sidebar.title("🤖 ML & DL Workbench")
    st.sidebar.caption(
        "Application minimaliste pour créer, entraîner, évaluer et déployer des modèles."
    )

    step = st.sidebar.radio("Navigation", SIDEBAR_STEPS, index=0)

    with st.sidebar.expander("État courant", expanded=False):
        st.write({
            "dataset": st.session_state.get("dataset_name"),
            "task": st.session_state.get("task_type"),
            "approche": st.session_state.get("approach"),
            "modèle": st.session_state.get("model_name"),
        })

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Astuce : progressez de haut en bas. Vous pouvez revenir aux étapes précédentes"
        " pour modifier la configuration.")

    return step


# -----------------------------------------------------------------------------
# Sections (filled by ui_components in later steps). For now, placeholders.
# -----------------------------------------------------------------------------

def section_load_data():
    st.header("1️⃣ Chargement des données")
    st.write(
        "Chargez un CSV/Excel, ou choisissez un dataset intégré (Iris, Wine, MNIST, etc.)."
    )

    try:
        if 'upload_data_ui' in globals():
            result = upload_data_ui()
            if result is not None:
                df, name = result
                st.session_state.df = df
                st.session_state.dataset_name = name
                reset_after_data_change()
        else:
            st.info("Le composant d'upload sera disponible à l'étape suivante.")
    except Exception as e:
        st.error(f"Erreur pendant le chargement des données : {e}")

    if st.session_state.df is not None:
        st.success("Données chargées ✔")
        st.dataframe(st.session_state.df.head())


def section_prepare_data():
    st.header("2️⃣ Préparation des données")
    st.write("Sélection de la cible/features, split train/test et prétraitements.")

    # ✅ Cas images (MNIST/CIFAR): on saute la préparation tabulaire
    if st.session_state.get("image_data") is not None:
        meta = st.session_state.get("image_meta", {})
        st.info(
            "Dataset **image** détecté (ex: MNIST/CIFAR).\n"
            f"Taille d'entrée: **{meta.get('input_shape')}**, classes: **{meta.get('num_classes')}**.\n"
            "Passez directement à l'étape **3️⃣ Sélection du modèle** pour choisir **CNN** puis **4️⃣ Entraînement**."
        )
        return

    if st.session_state.df is None:
        st.warning("Veuillez d'abord charger des données (Étape 1).")
        return

    try:
        if 'select_task_ui' in globals():
            select_task_ui()
        else:
            st.info("Les composants de préparation seront ajoutés à l'étape suivante.")
    except Exception as e:
        st.error(f"Erreur pendant la préparation des données : {e}")



def section_select_model():
    st.header("3️⃣ Sélection du modèle")
    st.write("Choisissez l'approche (Classique ou Deep Learning) et le modèle.")

    # ✅ Autoriser si tabulaire prêt OU si dataset image présent
    if st.session_state.X_train is None and st.session_state.get("image_data") is None:
        st.warning("Effectuez d'abord la préparation & le split (Étape 2) — "
                   "ou chargez un dataset **image** (MNIST/CIFAR).")
        return

    try:
        if 'model_selection_ui' in globals():
            model_selection_ui()
        else:
            st.info("Les composants de sélection du modèle seront ajoutés ensuite.")
    except Exception as e:
        st.error(f"Erreur pendant la sélection du modèle : {e}")




def section_training():
    st.header("4️⃣ Entraînement")
    st.write("Entraînez le modèle choisi, suivez les logs et l'historique.")

    if st.session_state.model_name is None and st.session_state.model is None:
        st.warning("Veuillez d'abord sélectionner un modèle (Étape 3).")
        return

    try:
        if 'training_ui' in globals():
            training_ui()
        else:
            st.info("Les composants d'entraînement seront ajoutés ensuite.")
    except Exception as e:
        st.error(f"Erreur pendant l'entraînement : {e}")


def section_evaluation():
    st.header("5️⃣ Évaluation")
    st.write("Consultez les métriques et visualisations associées.")

    if st.session_state.model is None:
        st.warning("Veuillez d'abord entraîner un modèle (Étape 4).")
        return

    try:
        if 'evaluation_ui' in globals():
            evaluation_ui()
        else:
            st.info("Les composants d'évaluation seront ajoutés ensuite.")
    except Exception as e:
        st.error(f"Erreur pendant l'évaluation : {e}")


def section_prediction():
    st.header("6️⃣ Prédiction")
    st.write("Faites des prédictions manuelles ou par fichier/image.")

    if st.session_state.model is None:
        st.warning("Veuillez d'abord entraîner un modèle (Étape 4).")
        return

    try:
        if 'prediction_ui' in globals():
            prediction_ui()
        else:
            st.info("Les composants de prédiction seront ajoutés ensuite.")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")


def section_export():
    st.header("7️⃣ Export & Réutilisation")
    st.write("Sauvegardez/rechargez le modèle et exportez un rapport.")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.model is not None:
            default_name = f"{st.session_state.model_name or 'model'}"
            fname = st.text_input("Nom du fichier modèle", value=default_name)
            if st.button("💾 Sauvegarder le modèle", use_container_width=True):
                try:
                    if 'save_model' in globals():
                        path = models_dir / fname
                        save_model(st.session_state.model, str(path))
                        st.success(f"Modèle sauvegardé dans {path}")
                    else:
                        st.info("La sauvegarde sera active après ajout de model_utils.py")
                except Exception as e:
                    st.error(f"Échec de la sauvegarde : {e}")
        else:
            st.info("Aucun modèle en mémoire à sauvegarder.")

    with col2:
        uploaded = st.file_uploader("📂 Recharger un modèle entraîné", type=["pkl", "h5"], key="reload_model")
        if uploaded is not None:
            tmp_path = models_dir / f"_tmp_{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            try:
                if 'load_model' in globals():
                    model = load_model(str(tmp_path))
                    st.session_state.model = model
                    st.success("Modèle rechargé ✔")
                else:
                    st.info("Le chargement sera actif après ajout de model_utils.py")
            except Exception as e:
                st.error(f"Échec du chargement : {e}")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    st.markdown("---")
    st.caption(
        "Optionnel : la génération d'un mini rapport sera ajoutée lors de l'étape\n"
        "d'évaluation (metrics + hyperparamètres + horodatage)."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    step = sidebar_nav()

    # Use tabs for quick navigation within the selected step (kept simple)
    if step == SIDEBAR_STEPS[0]:
        section_load_data()
    elif step == SIDEBAR_STEPS[1]:
        section_prepare_data()
    elif step == SIDEBAR_STEPS[2]:
        section_select_model()
    elif step == SIDEBAR_STEPS[3]:
        section_training()
    elif step == SIDEBAR_STEPS[4]:
        section_evaluation()
    elif step == SIDEBAR_STEPS[5]:
        section_prediction()
    elif step == SIDEBAR_STEPS[6]:
        section_export()


if __name__ == "__main__":
    main()
