import streamlit as st
import os
from ui.actions.upload import upload_decks
from ui.actions.compare import compare_uploaded_files
from ui.actions.save import save_uploaded_files 
from ui.actions.vecto_predict import run_vectorize_and_predict_ui
from ui.actions.display_results import display_prediction_results


# --- Chemins ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "data", "decks")
TRANSLATED_DIR = os.path.join(BASE_DIR, "data", "processed", "translated")

# --- Page config ---
st.set_page_config(page_title="Scoring App", page_icon="🎯", layout="centered")

# --- Session state ---
if "page" not in st.session_state:
    st.session_state.page = "menu"
if "comparison_done" not in st.session_state:
    st.session_state.comparison_done = False
if "saved_uploaded_files" not in st.session_state:
    st.session_state.saved_uploaded_files = False

# --- Navigation ---
def go_to(page_name):
    if page_name == "menu":
        st.session_state.clear()
        st.session_state.page = "menu"
    else:
        st.session_state.page = page_name

# --- Menu principal ---
if st.session_state.page == "menu":
    st.title("🎯 Scoring App")
    st.subheader("Choisissez une action :")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Entraîner les modèles"):
            go_to("train")
    with col2:
        if st.button("📊 Analyser un ou plusieurs decks"):
            go_to("analyze")

# --- Page entraînement ---
elif st.session_state.page == "train":
    st.title("🧠 Entraînement des modèles")
    st.write("👉 Ici tu pourras lancer l'entraînement de tes modèles.")
    if st.button("⬅️ Retour au menu principal"):
        go_to("menu")

# --- Page analyse ---
elif st.session_state.page == "analyze":
    st.title("📊 Analyse des decks")
    st.write("👉 Sélectionnez un ou plusieurs fichiers PDF à analyser.")

    uploaded_files = upload_decks()
    compare_uploaded_files(uploaded_files, TRANSLATED_DIR, DECKS_DIR)
    save_uploaded_files(uploaded_files, DECKS_DIR, TRANSLATED_DIR)

    # --- Étape : Vectorisation + Prédictions ---
    run_vectorize_and_predict_ui()
    if st.session_state.get("predictions_done", False):
        uploaded_saved_names = []
        for file in uploaded_files:
            original_name = file.name
            rename_key = f"rename_{original_name}"
            final_name = st.session_state.get(rename_key, original_name)
            uploaded_saved_names.append(final_name)

        display_prediction_results(uploaded_saved_names)

    if st.button("⬅️ Retour au menu principal"):
        go_to("menu")
