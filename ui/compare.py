import os
from difflib import SequenceMatcher
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.translate import translate_text

"""
Module de comparaison de fichiers PDF uploadés avec des fichiers TXT traduits existants.

Fonctionnalités :
- Extraction et traduction du texte des PDF uploadés.
- Calcul de la similarité entre le texte uploadé et le texte existant (SequenceMatcher).
- Affichage dans Streamlit avec des expanders pour visualiser le texte côté uploadé et côté existant.
- Vérification de l’existence du fichier dans le répertoire des decks et proposition de renommage automatique.

Paramètres :
- uploaded_files : liste de fichiers PDF uploadés via Streamlit.
- translated_dir : chemin du répertoire contenant les fichiers TXT traduits.
- decks_dir : chemin du répertoire des decks existants.

Effets :
- Met à jour `st.session_state.comparison_done` après la comparaison.
- Affiche des warnings et des suggestions de renommage si des fichiers existent déjà.
"""


def compare_uploaded_files(uploaded_files, translated_dir, decks_dir):
    """
    Compare les fichiers PDF uploadés avec les fichiers TXT traduits existants et affiche la similarité.

    Étapes principales :
    1. Vérifie que des fichiers ont été uploadés, sinon affiche un message d'information.
    2. Pour chaque fichier PDF :
       a. Vérifie si le fichier TXT traduit correspondant existe.
       b. Extrait et traduit le texte du PDF uploadé.
       c. Lit le texte existant depuis le fichier TXT.
       d. Calcule la similarité entre le texte uploadé et le texte existant.
       e. Affiche les textes dans deux colonnes via un expander Streamlit, avec la similarité en titre.
       f. Vérifie si un fichier du même nom existe déjà dans le répertoire des decks et propose un renommage automatique.
    3. Met à jour `st.session_state.comparison_done` à True une fois la comparaison terminée.

    Paramètres
    ----------
    uploaded_files : list
        Liste des fichiers PDF uploadés via Streamlit.
    translated_dir : str
        Chemin du répertoire contenant les fichiers TXT traduits.
    decks_dir : str
        Chemin du répertoire des decks existants.

    Effets
    -------
    - Affiche les résultats dans l'interface Streamlit.
    - Met à jour `st.session_state.comparison_done`.
    - Génère des suggestions de renommage si des fichiers existent déjà.
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
