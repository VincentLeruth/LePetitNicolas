# ui/train_ui.py
import os
import streamlit as st
import pandas as pd

from src.ml.domain.model_domain import train_domain
from src.ml.country.model_country import train_country
from src.ml.tech.model_tech import train_tech
from src.ml.resultat.model_result import train_result

from commite_github import commit_file_to_github

"""
Module Streamlit pour l'interface d'entraînement des modèles ML.

Fonctionnalités :
- Interface pour labelliser manuellement les fichiers PDF (decks) restants.
- Sélection des labels pour chaque axe : technologie, domaine, pays et résultat.
- Sauvegarde automatique des corrections dans labeled.csv.
- Gestion des decks restants dans st.session_state.
- Bouton pour entraîner tous les modèles lorsque tous les decks sont labellisés.
- Interface Streamlit avec boutons valider/ignorer et mise à jour en temps réel.
"""

# --- Chemins ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "..", "data", "decks")
LABELED_CSV = os.path.join(BASE_DIR, "..", "data", "labeled.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# --- Choix possibles pour chaque axe ---
DOMAINS = ["energy transition", "industrie 4.0", "new materials", "others"]
COUNTRIES = ["benelux", "france", "germany", "autres"]
TECHS = ["soft", "hard", "both"]
RESULTS = ["Unfavorable", "Very Unfavorable", "Interessant", "Out"]


def run_training_ui():
    """
    Interface Streamlit pour la labellisation des decks et l'entraînement des modèles.

    Étapes principales :
    1. Charge le fichier labeled.csv s'il existe, sinon crée un DataFrame vide.
    2. Liste tous les fichiers PDF dans le répertoire des decks.
    3. Met à jour st.session_state.remaining_decks avec les fichiers non encore labellisés.
    4. Si tous les decks sont labellisés :
       a. Affiche un message de succès.
       b. Propose un bouton pour entraîner tous les modèles.
    5. Si des decks restent à labelliser :
       a. Affiche le deck actuel.
       b. Initialise les valeurs par défaut à partir de st.session_state.corrections.
       c. Propose des listes déroulantes pour chaque axe (tech, domain, country, result).
       d. Bouton "Valider" : sauvegarde la correction dans la session et dans labeled.csv.
       e. Bouton "Ignorer" : passe le deck au suivant sans le sauvegarder.

    Effets
    -------
    - Met à jour labeled.csv avec les corrections validées.
    - Met à jour st.session_state.remaining_decks et st.session_state.corrections.
    - Permet d'entraîner les modèles ML une fois tous les decks labellisés.
    """
    
    st.subheader("🧠 Entraînement des modèles")

    # --- Charger labeled.csv ou créer DataFrame vide ---
    if os.path.exists(LABELED_CSV):
        labeled_df = pd.read_csv(LABELED_CSV, sep=";")
    else:
        labeled_df = pd.DataFrame(columns=["doc", "tech", "domain", "country", "result"])

    # --- Lister tous les decks PDF ---
    all_decks = [f for f in os.listdir(DECKS_DIR) if f.endswith(".pdf")]

    # --- Session state : decks restants à labelliser ---
    if "remaining_decks" not in st.session_state:
        st.session_state.remaining_decks = [f for f in all_decks if f not in labeled_df["doc"].tolist()]

    if not st.session_state.remaining_decks:
        st.success("✅ Tous les decks ont été labellisés !")
        if st.button("🧠 Entraîner tous les modèles"):
            st.info("⏳ Entraînement en cours... Cela peut prendre quelques minutes.")

            # --- Entraînement de chaque modèle ---
            train_domain()
            commit_file_to_github(os.path.join(MODELS_DIR, "domain_gb_model.joblib"),
                                  "models/domain_gb_model.joblib",
                                  "Mise à jour du modèle domain")

            train_country()
            commit_file_to_github(os.path.join(MODELS_DIR, "country_gb_model.joblib"),
                                  "models/country_gb_model.joblib",
                                  "Mise à jour du modèle country")

            train_tech()
            commit_file_to_github(os.path.join(MODELS_DIR, "tech_gb_model.joblib"),
                                  "models/tech_gb_model.joblib",
                                  "Mise à jour du modèle tech")

            train_result()
            commit_file_to_github(os.path.join(MODELS_DIR, "result_gb_model.joblib"),
                                  "models/result_gb_model.joblib",
                                  "Mise à jour du modèle result")

            st.success("🎉 Tous les modèles ont été entraînés et sauvegardés !")
        return

    # --- Deck actuel à corriger ---
    current_deck = st.session_state.remaining_decks[0]
    st.markdown(f"### 📄 {current_deck} (encore {len(st.session_state.remaining_decks)} decks à vérifier)")

    # --- Valeurs par défaut si déjà corrigé dans cette session ---
    if "corrections" not in st.session_state:
        st.session_state.corrections = {}

    default_vals = st.session_state.corrections.get(current_deck, {})
    tech_default = default_vals.get("tech", TECHS[0])
    domain_default = default_vals.get("domain", DOMAINS[0])
    country_default = default_vals.get("country", COUNTRIES[0])
    result_default = default_vals.get("result", RESULTS[0])

    # --- Listes déroulantes pour sélection des labels avec key unique ---
    tech = st.selectbox("🧠 Technologie", TECHS, index=TECHS.index(tech_default), key=f"tech_{current_deck}")
    domain = st.selectbox("🌍 Domaine", DOMAINS, index=DOMAINS.index(domain_default), key=f"domain_{current_deck}")
    country = st.selectbox("🏳️ Pays", COUNTRIES, index=COUNTRIES.index(country_default), key=f"country_{current_deck}")
    result = st.selectbox("🎯 Résultat", RESULTS, index=RESULTS.index(result_default), key=f"result_{current_deck}")

    col1, col2 = st.columns(2)

    # --- Bouton Valider : sauvegarde la correction ---
    with col1:
        if st.button(f"✅ Valider {current_deck}", key=f"valider_{current_deck}"):
            # Sauvegarder la correction dans la session
            st.session_state.corrections[current_deck] = {
                "tech": tech,
                "domain": domain,
                "country": country,
                "result": result
            }

            # Sauvegarde immédiate dans labeled.csv
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

            # Retirer le deck de la liste restante et passer au suivant
            st.session_state.remaining_decks.pop(0)
            st.rerun()

    # --- Bouton Ignorer : passe le deck au suivant sans sauvegarder ---
    with col2:
        if st.button(f"⏭ Ignorer {current_deck}", key=f"ignorer_{current_deck}"):
            st.warning(f"⚠️ {current_deck} ignoré pour le moment")
            st.session_state.remaining_decks.pop(0)
            st.rerun()
