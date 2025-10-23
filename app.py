import streamlit as st
import os
import difflib
from ui.display_results import afficher_resultat_deck
from ui.predictions import lancer_vectorisation_et_predictions
from ui.model_train_ui import run_model_training_ui
from ui.upload import extraire_texte  # ← ta fonction existante

# --- Définition des chemins ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "data", "decks")
TRANSLATED_DIR = os.path.join(BASE_DIR, "data", "processed", "translated")

# --- Initialisation du state ---
st.set_page_config(layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "menu_principal"
if "uploaded_file_objs" not in st.session_state:
    st.session_state.uploaded_file_objs = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "validation_done" not in st.session_state:
    st.session_state.validation_done = False
if "vectorisation_done" not in st.session_state:
    st.session_state.vectorisation_done = False
if "conflict_message" not in st.session_state:
    st.session_state.conflict_message = ""


# --- Fonctions utilitaires ---

def reset_analysis_state():
    st.session_state.uploaded_file_objs = None
    st.session_state.uploaded_files = []
    st.session_state.validation_done = False
    st.session_state.vectorisation_done = False
    st.session_state.conflict_message = ""


def next_free_name(basename, ext):
    """Renomme en basename_1.ext, basename_2.ext, ... selon ce qui existe."""
    i = 1
    while True:
        candidate = f"{basename}_{i}{ext}"
        if not os.path.exists(os.path.join(DECKS_DIR, candidate)):
            return candidate
        i += 1


def save_fileobj_to_decks(fileobj, target_name):
    """Sauvegarde l'objet file_uploader dans DECKS_DIR sous target_name."""
    path = os.path.join(DECKS_DIR, target_name)
    with open(path, "wb") as out_file:
        out_file.write(fileobj.getbuffer())
    return target_name


def load_existing_text(deck_name):
    """Charge le texte déjà existant associé à un deck (fichier .txt)."""
    txt_path = os.path.join(TRANSLATED_DIR, f"{os.path.splitext(deck_name)[0]}.txt")
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def compare_texts(existing_text: str, new_text: str) -> dict:
    """Compare deux textes et retourne métriques et extrait de diff."""
    similarity = difflib.SequenceMatcher(None, existing_text, new_text).ratio()
    diff = list(difflib.unified_diff(
        existing_text.splitlines(),
        new_text.splitlines(),
        lineterm="",
        n=5
    ))
    diff_excerpt = "\n".join(diff[:15]) if diff else "Aucune différence visible dans les 5 premières lignes."
    return {"similarity": similarity, "diff_excerpt": diff_excerpt}


# --- App principale ---

def main():
    st.title("📊 Plateforme d'analyse de decks et d'entraînement de modèles")

    if st.session_state.page == "menu_principal":
        st.subheader("Que souhaitez-vous faire ?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Analyser un ou plusieurs decks", use_container_width=True):
                reset_analysis_state()
                st.session_state.page = "analyse_decks"
        with col2:
            if st.button("🧠 Entraîner les modèles", use_container_width=True):
                st.session_state.page = "entrainement"

    elif st.session_state.page == "analyse_decks":
        st.header("📂 Analyse de decks")

        uploaded = st.file_uploader(
            "Téléversez un ou plusieurs fichiers PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader"
        )

        if uploaded:
            st.session_state.uploaded_file_objs = uploaded

        # Étape comparaison / validation
        if st.session_state.uploaded_file_objs and not st.session_state.validation_done:
            existing = []
            new = []
            for f in st.session_state.uploaded_file_objs:
                if os.path.exists(os.path.join(DECKS_DIR, f.name)):
                    existing.append(f.name)
                else:
                    new.append(f.name)

            if existing:
                st.warning("⚠️ Des fichiers existent déjà :")
                for deck_name in existing:
                    existing_text = load_existing_text(deck_name)
                    new_text = extraire_texte(f)
                    if existing_text and new_text:
                        comp = compare_texts(existing_text, new_text)
                        with st.expander(f"🔎 Comparaison pour {deck_name}", expanded=False):
                            st.write(f"**Taux de similarité :** {comp['similarity']*100:.2f}%")
                            st.text_area("Différences détectées :", comp["diff_excerpt"], height=180)
                    else:
                        st.info(f"Impossible de comparer {deck_name} (texte manquant).")

                action = st.radio(
                    "Que souhaitez-vous faire ?",
                    ["Écraser", "Renommer", "Annuler"],
                    key="conflict_action"
                )

                if st.button("✅ Valider le choix", key="validate_conflict"):
                    if action == "Annuler":
                        st.info("Action annulée. Retour au menu principal.")
                        reset_analysis_state()
                        st.session_state.page = "menu_principal"
                        return

                    saved = []
                    for f in st.session_state.uploaded_file_objs:
                        base, ext = os.path.splitext(f.name)
                        if f.name in existing:
                            target = f.name if action == "Écraser" else next_free_name(base, ext)
                        else:
                            target = f.name
                        saved.append(save_fileobj_to_decks(f, target))

                    st.session_state.uploaded_files = saved
                    st.session_state.validation_done = True
                    st.success(f"✅ Fichiers enregistrés ({action}).")

            else:
                # Aucun doublon : sauvegarde directe
                if st.button("✅ Valider l'import des fichiers", key="validate_no_conflict"):
                    saved = []
                    for f in st.session_state.uploaded_file_objs:
                        saved.append(save_fileobj_to_decks(f, f.name))
                    st.session_state.uploaded_files = saved
                    st.session_state.validation_done = True
                    st.success("✅ Tous les fichiers sont nouveaux et ont été enregistrés.")

        # Étape post-validation
        if st.session_state.validation_done:
            if not st.session_state.vectorisation_done:
                if st.button("🚀 Lancer la vectorisation et les prédictions", key="launch_vector"):
                    lancer_vectorisation_et_predictions(st.session_state.uploaded_files)
                    st.session_state.vectorisation_done = True
                    st.success("Vectorisation et prédictions terminées.")

        if st.session_state.vectorisation_done:
            st.markdown("---")
            st.subheader("📈 Résultats d’analyse")
            for name in st.session_state.uploaded_files:
                with st.expander(f"Résultat : {name}", expanded=False):
                    afficher_resultat_deck(name)

        if st.button("⬅️ Retour au menu principal", key="back_to_menu_from_analysis"):
            reset_analysis_state()
            st.session_state.page = "menu_principal"

    elif st.session_state.page == "entrainement":
        st.header("🧠 Entraînement des modèles")
        run_model_training_ui()
        if st.button("⬅️ Retour au menu principal", key="back_to_menu_from_training"):
            st.session_state.page = "menu_principal"


if __name__ == "__main__":
    main()
