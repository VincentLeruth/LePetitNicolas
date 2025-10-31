import os
import streamlit as st
import pandas as pd

from src.vectorisation.vectorize_text import vectorize_text
from src.ml.domain.model_domain import train_domain
from src.ml.country.model_country import train_country
from src.ml.tech.model_tech import train_tech
from src.ml.resultat.model_result import train_result
from streamlit_pdf_viewer import pdf_viewer  
from synchro_github import sync_repo

# --- Chemins ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "..", "data", "decks")
LABELED_CSV = os.path.join(BASE_DIR, "..", "data", "labeled.csv")

# --- Choix possibles pour chaque axe ---
DOMAINS = ["energy transition", "industrie 4.0", "new materials", "others"]
COUNTRIES = ["benelux", "france", "germany", "other"]
TECHS = ["soft", "hard", "both"]
RESULTS = ["Unfavorable", "Very Unfavorable", "Interesting", "Out"]


def run_training_ui():
    """
    Interface Streamlit pour la labellisation des decks PDF et l'entraînement
    des modèles ML (domain, country, tech, result). 
    
    Fonctionnalités :
    - Affiche un formulaire de labellisation pour chaque deck restant.
    - Permet de valider ou ignorer un deck.
    - Entraîne tous les modèles une fois tous les decks labellisés.
    - Synchronise le repo GitHub après entraînement.

    La fonction gère les états via `st.session_state` :
    - `remaining_decks` : liste des decks encore à labelliser
    - `corrections` : labellisations temporaires pour chaque deck
    - `pushed_after_training` : flag pour éviter les push GitHub répétés
    """

    # --- Style pour étendre la largeur de la page ---
    st.markdown("""
        <style>
        .block-container {
            max-width: 95% !important;
            padding-left: 2% !important;
            padding-right: 2% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("🧠 Entraînement des modèles")

    # --- Charger ou créer le fichier labeled.csv ---
    if os.path.exists(LABELED_CSV):
        labeled_df = pd.read_csv(LABELED_CSV, sep=";")
    else:
        labeled_df = pd.DataFrame(columns=["doc", "tech", "domain", "country", "result"])

    # --- Lister tous les decks PDF ---
    all_decks = [f for f in os.listdir(DECKS_DIR) if f.endswith(".pdf")]

    # --- Initialiser la liste des decks restants ---
    if "remaining_decks" not in st.session_state:
        st.session_state.remaining_decks = [f for f in all_decks if f not in labeled_df["doc"].tolist()]

    # --- Cas où tous les decks sont labellisés ---
    if not st.session_state.remaining_decks:
        st.success("✅ Tous les decks ont été labellisés !")
        
        # --- Bouton pour entraîner tous les modèles ---
        if st.button("🧠 Entraîner tous les modèles"):
            
            # Spinner pendant entraînement et sync
            with st.spinner("⏳ Entraînement en cours..."):
                vectorize_text()
                train_domain()
                train_country()
                train_tech()
                train_result()

            st.success("🎉 Tous les modèles ont été entraînés  !")
        

    # --- Deck courant ---
    current_deck = st.session_state.remaining_decks[0]
    st.markdown(f"### 📄 {current_deck} (encore {len(st.session_state.remaining_decks)} à traiter)")

    # --- Layout horizontal : formulaire à gauche / PDF à droite ---
    col_form, col_pdf = st.columns([1.5, 3.5])

    with col_form:
        # --- Pré-remplissage des valeurs si déjà saisies ---
        if "corrections" not in st.session_state:
            st.session_state.corrections = {}

        default_vals = st.session_state.corrections.get(current_deck, {})
        tech_default = default_vals.get("tech", TECHS[0])
        domain_default = default_vals.get("domain", DOMAINS[0])
        country_default = default_vals.get("country", COUNTRIES[0])
        result_default = default_vals.get("result", RESULTS[0])

        # --- Sélecteurs pour labellisation ---
        tech = st.selectbox("🧠 Technologie", TECHS, index=TECHS.index(tech_default))
        domain = st.selectbox("🌍 Domaine", DOMAINS, index=DOMAINS.index(domain_default))
        country = st.selectbox("🏳️ Pays", COUNTRIES, index=COUNTRIES.index(country_default))
        result = st.selectbox("🎯 Resultat", RESULTS, index=RESULTS.index(result_default))

        # --- Boutons Valider / Ignorer ---
        btn_col1, btn_col2 = st.columns(2)

        # --- Valider deck ---
        with btn_col1:
            if st.button(f"✅ Valider {current_deck}"):
                st.session_state.corrections[current_deck] = {
                    "tech": tech,
                    "domain": domain,
                    "country": country,
                    "result": result
                }
                new_row = pd.DataFrame([{
                    "doc": current_deck,
                    "tech": tech,
                    "domain": domain,
                    "country": country,
                    "result": result
                }])
                labeled_df = pd.concat([labeled_df, new_row], ignore_index=True)
                labeled_df.to_csv(LABELED_CSV, sep=";", index=False)
                st.success(f"✅ {current_deck} ajouté à labeled.csv")
               
                st.session_state.remaining_decks.pop(0)
                st.rerun()

        # --- Ignorer deck ---
        with btn_col2:
            if st.button(f"⏭ Ignorer {current_deck}"):
                st.warning(f"⚠️ {current_deck} ignoré temporairement.")
                st.session_state.remaining_decks.pop(0)
                st.rerun()

    with col_pdf:
        # --- Affichage PDF ---
        pdf_path = os.path.join(DECKS_DIR, current_deck)
        st.markdown("### 👀 Aperçu du deck")
        if os.path.exists(pdf_path):
            pdf_viewer(pdf_path, width="100%", height=800, zoom_level=1.0)
        else:
            st.warning("⚠️ Fichier PDF introuvable dans le dossier 'data/decks'.")
