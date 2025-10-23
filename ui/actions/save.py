import os
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.translate import translate_text


def save_uploaded_files(uploaded_files, decks_dir, translated_dir):
    """
    Sauvegarde tous les fichiers uploadés dans le dossier decks
    et génère les fichiers TXT traduits dans translated_dir.
    Met à jour st.session_state.saved_uploaded_files à True après sauvegarde.
    Le bouton disparaît après la première sauvegarde.
    """
    if not uploaded_files:
        st.warning("Aucun fichier à sauvegarder.")
        return

    # Initialisation du flag si inexistant
    if "saved_uploaded_files" not in st.session_state:
        st.session_state.saved_uploaded_files = False

    # Si déjà sauvegardé → on affiche juste un message
    if st.session_state.saved_uploaded_files:
        st.info("✅ Les fichiers ont déjà été sauvegardés.")
        return

    # Affichage du bouton seulement si pas encore sauvegardé
    if st.button("💾 Sauvegarder tous les fichiers uploadés"):
        for file in uploaded_files:
            original_name = file.name
            rename_key = f"rename_{original_name}"
            save_name = st.session_state.get(rename_key, original_name)
   
            # --- Sauvegarde du PDF ---
            save_path = os.path.join(decks_dir, save_name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            # --- Génération du TXT traduit ---
            txt_path = os.path.join(translated_dir, os.path.splitext(save_name)[0] + ".txt")
            uploaded_text = translate_text(extract_text_from_pdf(file))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(uploaded_text)

        # Flag → sauvegarde effectuée
        st.session_state.saved_uploaded_files = True
        st.success("✅ Tous les fichiers uploadés et leurs TXT traduits ont été sauvegardés.")
        st.rerun()

