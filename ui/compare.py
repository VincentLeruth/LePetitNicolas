import os
from difflib import SequenceMatcher
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.translate import translate_text

def compare_uploaded_files(uploaded_files, translated_dir, decks_dir):
    """
    Compare les fichiers uploadés avec les fichiers TXT traduits existants
    et affiche la similarité et les expander pour la visualisation.
    """
    if not uploaded_files:
        st.info("⬆️ Uploadez des fichiers PDF pour lancer la comparaison.")
        st.session_state.comparison_done = False
        return

    for file in uploaded_files:
        filename = file.name
        txt_path = os.path.join(translated_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(txt_path):
            st.warning(f"Aucun fichier TXT traduit trouvé pour `{filename}`")
            continue

        uploaded_text = translate_text(extract_text_from_pdf(file))
        with open(txt_path, "r", encoding="utf-8") as f:
            existing_text = f.read()

        similarity = SequenceMatcher(None, uploaded_text, existing_text).ratio() * 100

        with st.expander(f"Comparaison pour : {filename} — Similarité : {similarity:.2f}%"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploadé")
                st.text(uploaded_text[:1000] + ("..." if len(uploaded_text) > 1000 else ""))
            with col2:
                st.subheader("Existant (TXT)")
                st.text(existing_text[:1000] + ("..." if len(existing_text) > 1000 else ""))

            # Vérification si le fichier existe déjà dans decks
            existing_deck_path = os.path.join(decks_dir, filename)
            if os.path.exists(existing_deck_path):
                st.markdown(
                    "**<span style='color:red'>⚠️ Ce fichier existe déjà. "
                    "Si vous ne le supprimez pas, vous pouvez renommer ce deck.</span>**",
                    unsafe_allow_html=True
                )
                base_name, ext = os.path.splitext(filename)
                version = 1
                while os.path.exists(os.path.join(decks_dir, f"{base_name}_v{version}{ext}")):
                    version += 1
                new_name = f"{base_name}_v{version}{ext}"
                st.text_input("Renommer ce deck :", value=new_name, key=f"rename_{filename}")

    st.session_state.comparison_done = True
