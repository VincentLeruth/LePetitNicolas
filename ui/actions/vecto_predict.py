import streamlit as st

# Import de tes modules existants
from src.vectorisation.vectorize_text import vectorize_text
from src.ml.domain.predict_domain import predict_domain
from src.ml.country.predict_country import predict_country
from src.ml.tech.predict_tech import predict_tech
from src.ml.resultat.predict_resultat import predict_resultat


def run_vectorize_and_predict_ui():
    """
    Interface Streamlit pour enchaîner la vectorisation TF-IDF et les prédictions.
    Les boutons disparaissent après exécution.
    """
    st.subheader("🧮 Vectorisation + Prédictions automatiques")

    # Vérifier que les fichiers sont sauvegardés avant de lancer
    if not st.session_state.get("saved_uploaded_files", False):
        st.info("➡️ Veuillez d'abord sauvegarder les fichiers uploadés avant de lancer la vectorisation et les prédictions.")
        return

    # Initialisation des flags si nécessaire
    if "vectorization_done" not in st.session_state:
        st.session_state.vectorization_done = False
    if "predictions_done" not in st.session_state:
        st.session_state.predictions_done = False

    # --- Étape 1 : Vectorisation ---
    if not st.session_state.vectorization_done:
        if st.button("⚙️ Lancer la vectorisation TF-IDF"):
            with st.spinner("Vectorisation en cours..."):
                try:
                    vectorize_text()
                    st.session_state.vectorization_done = True
                    st.success("✅ Vectorisation terminée avec succès ! Les vecteurs ont été sauvegardés.")
                    st.rerun()  # 🔁 Recharge la page pour cacher le bouton
                except Exception as e:
                    st.error(f"❌ Erreur pendant la vectorisation : {e}")
            st.stop()

    # --- Étape 2 : Prédictions ---
    elif st.session_state.vectorization_done and not st.session_state.predictions_done:
        st.write("✅ Les vecteurs TF-IDF sont prêts. Vous pouvez maintenant lancer les prédictions.")
        if st.button("🤖 Lancer les prédictions sur tous les modèles"):
            with st.spinner("Prédictions en cours..."):
                try:
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
