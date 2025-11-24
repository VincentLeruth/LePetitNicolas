import os
import streamlit as st
import pandas as pd

"""
Module d'affichage des résultats de prédictions pour des fichiers uploadés.

Nouvelle logique :
- Lecture d’un seul fichier : labeled.csv
- Extraction pour chaque fichier des colonnes : Domaine, Pays, Technologie, Résultat
- Vérification des critères autorisés
- Affichage sous forme de cartes repliables Streamlit
"""

BASE_DIR = os.path.dirname(__file__)

# --- Définir les critères ---
ALLOWED_COUNTRIES = ["benelux", "france", "germany"]
ALLOWED_DOMAINS = ["energy transition", "industry 4.0", "new materials"]


def display_prediction_results(uploaded_saved_names):
    """
    Affiche les résultats des prédictions à partir du fichier unique labeled.csv.
    """

    st.subheader("📊 Résultats des prédictions par fichier uploadé")

    # --- Chemin vers le fichier centralisé ---
    labeled_path = os.path.join(BASE_DIR, "..", "data", "labeled.csv")

    # --- Chargement du fichier ---
    try:
        df = pd.read_csv(labeled_path, sep=";")
    except FileNotFoundError:
        st.error("❌ Impossible de charger labeled.csv dans le répertoire de prédictions.")
        return

    # --- Vérification présence colonnes requises ---
    required_cols = ["doc", "tech", "domain", "country", "resultat"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ Colonnes manquantes dans labeled.csv : {', '.join(missing)}")
        return

    # --- Renommage propre ---
    df.rename(columns={
        "doc": "Fichier",
        "predicted_domain": "Domaine",
        "predicted_country": "Pays",
        "predicted_tech": "Technologie",
        "predicted_resultat": "Resultat"
    }, inplace=True)

    # --- Filtrer uniquement les fichiers réellement sauvegardés ---
    df = df[df["Fichier"].isin(uploaded_saved_names)]

    # --- Affichage des résultats ---
    for _, row in df.iterrows():
        country_out = row["Pays"].lower() not in ALLOWED_COUNTRIES
        domain_out = row["Domaine"].lower() not in ALLOWED_DOMAINS

        markers = ""
        if country_out:
            markers += "❌ Pays OUT  "
        if domain_out:
            markers += "❌ Domaine OUT"

        bg_color = "#f8d7da" if country_out or domain_out else "#f8f9fa"

        with st.expander(f"📄 {row['Fichier']} {markers}"):
            st.markdown(
                f"""
                <div style='background-color:{bg_color};padding:1rem;border-radius:10px;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.1);'>
                    <p><b>🌍 Domaine :</b> {row['Domaine']}</p>
                    <p><b>🏳️ Pays :</b> {row['Pays']}</p>
                    <p><b>🧠 Technologie (Hardware / Software / Both) :</b> {row['Technologie']}</p>
                    <p><b>🎯 Résultat :</b> {row['Resultat']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
