import os
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

# --- Définir les critères ---
ALLOWED_COUNTRIES = ["benelux", "france", "germany"]
ALLOWED_DOMAINS = ["energy transition", "industrie 4.0", "new materials"]

def display_prediction_results(uploaded_saved_names):
    """
    Affiche les résultats des prédictions par fichier (deck) réellement sauvegardé
    sous forme de cartes repliables avec indicateur OUT si hors critères.
    """
    
    processed_dir = os.path.join(BASE_DIR, "..", "..", "output", "predictions")
    st.subheader("📊 Résultats des prédictions par fichier uploadé")

    # --- Fichiers nécessaires ---
    files_needed = {
        "domain": os.path.join(processed_dir, "tfidf_vectors_with_domain_predictions.csv"),
        "country": os.path.join(processed_dir, "tfidf_vectors_with_country_predictions.csv"),
        "tech": os.path.join(processed_dir, "tfidf_vectors_with_tech_predictions.csv"),
        "resultat": os.path.join(processed_dir, "tfidf_vectors_with_resultat_predictions.csv"),
    }

    # --- Chargement des fichiers ---
    dfs = {key: pd.read_csv(path, sep=";") for key, path in files_needed.items()}

    # --- Fusion des résultats ---
    merged = dfs["domain"][["doc", "predicted_domain"]].copy()
    merged = merged.merge(dfs["country"][["doc", "predicted_country"]], on="doc", how="left")
    merged = merged.merge(dfs["tech"][["doc", "predicted_tech"]], on="doc", how="left")
    merged = merged.merge(dfs["resultat"][["doc", "predicted_resultat"]], on="doc", how="left")

    merged.rename(columns={
        "doc": "Fichier",
        "predicted_domain": "Domaine",
        "predicted_country": "Pays",
        "predicted_tech": "Technologie",
        "predicted_resultat": "Résultat"
    }, inplace=True)

    # --- Ne garder que les fichiers réellement sauvegardés ---
    merged = merged[merged["Fichier"].isin(uploaded_saved_names)]

    # --- Affichage sous forme de cartes repliables ---
    for _, row in merged.iterrows():
        country_out = row["Pays"].lower() not in ALLOWED_COUNTRIES
        domain_out = row["Domaine"].lower() not in ALLOWED_DOMAINS
        out_markers = ""
        if country_out:
            out_markers += "❌ Pays OUT  "
        if domain_out:
            out_markers += "❌ Domaine OUT"

        # Option : couleur rouge clair si OUT
        bg_color = "#f8d7da" if country_out or domain_out else "#f8f9fa"

        with st.expander(f"📄 {row['Fichier']} {out_markers}"):
            st.markdown(
                f"""
                <div style='background-color:{bg_color};padding:1rem;border-radius:10px;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.1);'>
                    <p><b>🌍 Domaine :</b> {row['Domaine']}</p>
                    <p><b>🏳️ Pays :</b> {row['Pays']}</p>
                    <p><b>🧠 Technologie :</b> {row['Technologie']}</p>
                    <p><b>🎯 Résultat :</b> {row['Résultat']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
