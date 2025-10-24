import os
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.translate import translate_text

def save_uploaded_files(uploaded_files, decks_dir, translated_dir):
    """
    Sauvegarde tous les fichiers uploadés dans decks_dir
    et génère les fichiers TXT traduits dans translated_dir.
    Met à jour st.session_state.uploaded_files_saved_names
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

            # --- Sauvegarde PDF ---
            save_path = os.path.join(decks_dir, final_name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            # --- Génération TXT traduit ---
            txt_path = os.path.join(translated_dir, os.path.splitext(final_name)[0] + ".txt")
            uploaded_text = translate_text(extract_text_from_pdf(file))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(uploaded_text)

            saved_files_names.append(final_name)

        # --- Mise à jour session state ---
        st.session_state.saved_uploaded_files = True
        st.session_state.uploaded_files_saved_names = saved_files_names

        st.success("✅ Tous les fichiers uploadés et leurs TXT traduits ont été sauvegardés.")
        st.rerun()
