import os
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.translate import translate_text
from commite_github import commit_file_to_github

"""
Module de sauvegarde des fichiers PDF uploadés et génération des fichiers TXT traduits.

Fonctionnalités :
- Sauvegarde des fichiers PDF uploadés dans le répertoire des decks.
- Extraction et traduction du texte des PDF pour générer des fichiers TXT traduits.
- Commit unique de tous les fichiers sauvegardés vers GitHub pour réduire le temps.
- Mise à jour de `st.session_state` avec les noms des fichiers sauvegardés.
- Bouton Streamlit pour déclencher la sauvegarde.
"""


def commit_all_files(saved_files, base_dir="data"):
    """
    Commit tous les fichiers d'une liste vers GitHub.

    Parameters
    ----------
    saved_files : list
        Liste des chemins complets des fichiers à commit.
    base_dir : str
        Chemin de base pour le commit GitHub (relatif au repo).

    Effets
    -------
    - Commit chaque fichier vers GitHub.
    - Affiche un message Streamlit après chaque commit.
    """
    st.info("🔄 Commit de tous les fichiers vers GitHub en cours...")

    for fpath in saved_files:
        if os.path.isfile(fpath):
            rel_path = os.path.join(base_dir, os.path.basename(fpath))
            commit_file_to_github(fpath, rel_path, f"Mise à jour : {os.path.basename(fpath)}")
            st.success(f"✅ {os.path.basename(fpath)} commit avec succès")

    st.success("🎉 Tous les fichiers ont été commit vers GitHub !")


def save_uploaded_files(uploaded_files, decks_dir, translated_dir):
    """
    Sauvegarde tous les fichiers uploadés et génère leurs fichiers TXT traduits.

    Étapes principales :
    1. Vérifie que des fichiers ont été uploadés.
    2. Si les fichiers ont déjà été sauvegardés, ne fait rien.
    3. Si le bouton Streamlit de sauvegarde est cliqué :
       a. Sauvegarde chaque PDF.
       b. Extrait et traduit le texte en TXT.
       c. Met à jour la liste des fichiers sauvegardés.
    4. Commit **tous les fichiers** à GitHub en une seule fois pour réduire le temps.
    5. Met à jour `st.session_state` et affiche un message de succès.

    Parameters
    ----------
    uploaded_files : list
        Liste des fichiers PDF uploadés via Streamlit.
    decks_dir : str
        Répertoire pour sauvegarder les PDF.
    translated_dir : str
        Répertoire pour sauvegarder les TXT traduits.
    """

    if not uploaded_files:
        st.warning("Aucun fichier à sauvegarder.")
        return

    if st.session_state.get("saved_uploaded_files", False):
        st.info("✅ Les fichiers ont déjà été sauvegardés.")
        return

    saved_files_names = []

    if st.button("💾 Sauvegarder tous les fichiers uploadés"):
        for file in uploaded_files:
            original_name = file.name
            rename_key = f"rename_{original_name}"
            final_name = st.session_state.get(rename_key, original_name)

            # --- Sauvegarde du fichier PDF ---
            save_path = os.path.join(decks_dir, final_name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            # --- Extraction et traduction du texte, puis sauvegarde en TXT ---
            txt_path = os.path.join(translated_dir, os.path.splitext(final_name)[0] + ".txt")
            uploaded_text = translate_text(extract_text_from_pdf(file))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(uploaded_text)

            saved_files_names.append(final_name)

        # --- Commit unique de tous les fichiers ---
        saved_files_paths = [os.path.join(decks_dir, f) for f in saved_files_names] + \
                            [os.path.join(translated_dir, os.path.splitext(f)[0] + ".txt") for f in saved_files_names]

        commit_all_files(saved_files_paths)

        # --- Mise à jour de st.session_state ---
        st.session_state.saved_uploaded_files = True
        st.session_state.uploaded_files_saved_names = saved_files_names

        st.success("✅ Tous les fichiers uploadés et leurs TXT traduits ont été sauvegardés.")
        st.rerun()
