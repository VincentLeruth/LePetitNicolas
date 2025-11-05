import streamlit as st
import os
import base64

# --- Import des modules UI existants ---
from ui.upload import upload_decks
from ui.compare import compare_uploaded_files
from ui.save import save_uploaded_files 
from ui.vecto_predict import run_vectorize_and_predict_ui
from ui.display_results import display_prediction_results
from ui.train import run_training_ui

from synchro_github import sync_repo

# --- Chemins ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "data", "decks")
TRANSLATED_DIR = os.path.join(BASE_DIR, "data", "processed", "translated")


# --- Configuration page ---
st.set_page_config(page_title="Le petit Nicolas", 
                   page_icon="🎯", 
                   layout="centered")


# --- Initialisation session state ---
if "page" not in st.session_state:
    st.session_state.page = "menu"
if "comparison_done" not in st.session_state:
    st.session_state.comparison_done = False
if "saved_uploaded_files" not in st.session_state:
    st.session_state.saved_uploaded_files = False

# --- Fonction de navigation entre pages ---
def go_to(page_name):
    """
    Change la page active et recharge l'interface.

    Parameters
    ----------
    page_name : str
        Nom de la page à afficher ("menu", "train", "analyze").
    """
    if page_name == "menu":
        st.session_state.clear()
        st.session_state.page = "menu"
    else:
        st.session_state.page = page_name
    st.rerun()

# --- Menu principal ---
if st.session_state.page == "menu":
    st.title("🎯 Le petit Nicolas")
    st.subheader("Choisissez une action :")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Entraîner les modèles (avec )" + str(len(os.listdir(DECKS_DIR)))):
            sync_repo(BASE_DIR, push=False)
            go_to("train")
    with col2:
        if st.button("📊 Analyser un ou plusieurs decks"):
            sync_repo(BASE_DIR, push=False)
            go_to("analyze")

# --- Page entraînement ---
elif st.session_state.page == "train":
    st.title("✅ Vérifications et Entraînement des modèles")
    st.write("👉 Ici tu peux labelliser les decks non traités et lancer l'entraînement des modèles.")

    run_training_ui()
    

    # --- Bouton retour au menu ---
    st.markdown("---")
    if st.button("⬅️ Retour au menu principal"):
        sync_repo(BASE_DIR, push=True)
        go_to("menu")


# --- Page analyse ---
elif st.session_state.page == "analyze":
    st.title("📊 Analyse des decks")
    st.write("👉 Sélectionnez un ou plusieurs fichiers PDF à analyser.")

    # --- Upload des fichiers ---
    uploaded_files = upload_decks()

    # --- Comparaison avec TXT traduits existants si fichiers non sauvegardés ---
    if not st.session_state.get("saved_uploaded_files", False):
        compare_uploaded_files(uploaded_files, TRANSLATED_DIR, DECKS_DIR)

    # --- Sauvegarde des fichiers uploadés et génération TXT ---
    saved_files = save_uploaded_files(uploaded_files, DECKS_DIR, TRANSLATED_DIR)

    # --- Vectorisation TF-IDF et prédictions automatiques ---
    run_vectorize_and_predict_ui()

    # --- Affichage des résultats si prédictions effectuées ---
    if st.session_state.get("predictions_done", False):
        saved_files_names = st.session_state.get("uploaded_files_saved_names", [])
        if saved_files_names:
            display_prediction_results(saved_files_names)

    # --- Sélection d'un deck via sidebar pour affichage spécifique ---
    deck_files = [f for f in os.listdir(DECKS_DIR) if f.lower().endswith(".pdf")]
    selected_file = st.sidebar.selectbox("📄 Sélectionnez un deck pour voir ses résultats", [""] + deck_files)
    st.sidebar.write("⚠️ Le résultat est à retrouver en bas de la page principal dans : \n\n Résultats des prédictions par fichier uploadé.")

    if selected_file:
        display_prediction_results(selected_file.split(sep=None, maxsplit=-1))

    # --- Bouton retour menu ---
    if st.button("⬅️ Enregistrer et retourner au menu principal"):
        if not st.session_state.get("pushed_after_analysis", False):
            sync_repo(BASE_DIR, push=True)
            st.session_state.pushed_after_analysis = True
        go_to("menu")

# --- Footer avec logo ---
logo_path = os.path.join(os.path.dirname(__file__), "Industrya_logo.jpg")
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode("utf-8")

st.markdown(
    f"""
    <style>
    /* Footer fixe en bas à gauche */
    .footer-container {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 20px;
        font-size: 14px;
        color: #555;
        border-top: 1px solid #ddd;
        z-index: 999;
    }}

    .footer-left img {{
        height: 60px;
        margin-right: 10px;
    }}

    .footer-right {{
        text-align: right;
        font-style: italic;
    }}
    </style>

    <div class="footer-container">
        <div class="footer-left">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo">
        </div>
        <div class="footer-right">
            © 2025 Scoring App — Tous droits réservés à Industrya Fund - Développé par Nicolas CB
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
