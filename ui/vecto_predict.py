import streamlit as st
import os

# Import des modules existants pour vectorisation et prédictions
from src.vectorisation.vectorize_text import vectorize_text
from src.ml.domain.predict_domain import predict_domain
from src.ml.country.predict_country import predict_country
from src.ml.tech.predict_tech import predict_tech
from src.ml.resultat.predict_resultat import predict_resultat

"""
Module Streamlit pour la vectorisation TF-IDF et les prédictions automatiques.

Fonctionnalités :
- Vérifie que les fichiers uploadés ont été sauvegardés avant de lancer la vectorisation.
- Lance la vectorisation TF-IDF sur les fichiers PDF sauvegardés.
- Effectue les prédictions pour tous les modèles (domain, country, tech, result) après vectorisation.
- Les boutons disparaissent une fois chaque étape terminée.
- Gestion de l'état via `st.session_state` pour éviter les doublons et suivre la progression.
"""

BASE_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "tfidf_vectors.csv")
PRED_DIR = os.path.join(BASE_DIR, "..", "output", "predictions")

def run_vectorize_and_predict_ui():
    """
    Interface Streamlit pour enchaîner la vectorisation TF-IDF et les prédictions.

    Étapes principales :
    1. Vérifie que les fichiers uploadés ont été sauvegardés.
    2. Initialise les flags `vectorization_done` et `predictions_done` dans st.session_state.
    3. Étape 1 : Vectorisation
       a. Affiche un bouton pour lancer la vectorisation TF-IDF.
       b. Met à jour `st.session_state.vectorization_done` après succès.
       c. Recharge la page pour masquer le bouton après exécution.
    4. Étape 2 : Prédictions
       a. Affiche un bouton pour lancer les prédictions sur tous les modèles.
       b. Met à jour `st.session_state.predictions_done` après succès.
       c. Recharge la page pour masquer le bouton après exécution.
    5. Étape finale : si vectorisation et prédictions déjà effectuées, affiche un message de confirmation.

    Effets
    -------
    - Exécute `vectorize_text()` pour générer les vecteurs TF-IDF.
    - Exécute `predict_domain()`, `predict_country()`, `predict_tech()`, `predict_resultat()`.
    - Met à jour `st.session_state` pour suivre l'avancement.
    - Affiche des messages Streamlit d'information, de succès ou d'erreur.
    """
    
    st.subheader("🧮 Vectorisation + Prédictions automatiques")

    # --- Vérifier que les fichiers sont sauvegardés avant de lancer ---
    if not st.session_state.get("saved_uploaded_files", False):
        st.info("➡️ Veuillez d'abord sauvegarder les fichiers uploadés avant de lancer la vectorisation et les prédictions.")
        return

    # --- Initialisation des flags si nécessaire ---
    if "vectorization_done" not in st.session_state:
        st.session_state.vectorization_done = False
    if "predictions_done" not in st.session_state:
        st.session_state.predictions_done = False

    # --- Étape 1 : Vectorisation ---
    if not st.session_state.vectorization_done and st.session_state.get("saved_uploaded_files", False):
        if st.button("⚙️ Lancer la vectorisation TF-IDF"):
            with st.spinner("Vectorisation en cours..."):
                try:
                    # ⚡ On met le flag à True avant le commit pour éviter le "retour arrière"
                    st.session_state.vectorization_done = True

                    # Exécution de la vectorisation
                    vectorize_text()
                    st.success("✅ Vectorisation terminée avec succès ! Les vecteurs ont été sauvegardés.")
                    st.rerun()  # Recharge la page pour cacher le bouton
                except Exception as e:
                    st.session_state.vectorization_done = False  # Reset si erreur
                    st.error(f"❌ Erreur pendant la vectorisation : {e}")
            st.stop()

    # --- Étape 2 : Prédictions ---
    elif st.session_state.vectorization_done and not st.session_state.predictions_done:
        st.write("✅ Les vecteurs TF-IDF sont prêts. Vous pouvez maintenant lancer les prédictions.")
        if st.button("🤖 Lancer les prédictions sur tous les modèles"):
            with st.spinner("Prédictions en cours..."):
                try:
                   
                    os.makedirs(PRED_DIR, exist_ok=True)

                    predict_domain()
                    predict_country()
                    predict_tech()
                    predict_resultat()
                           
                    st.session_state.predictions_done = True
                    st.success("🎯 Toutes les prédictions ont été effectuées avec succès !")
                    st.rerun()  # 🔁 Recharge pour masquer le bouton de prédiction
                except Exception as e:
                    st.error(f"❌ Erreur pendant les prédictions : {e}")
            st.stop()

    # --- Étape finale ---
    elif st.session_state.vectorization_done and st.session_state.predictions_done:
        st.success("✅ Vectorisation et prédictions déjà effectuées.")
