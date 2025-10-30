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
- Mise à jour de `st.session_state` avec les noms des fichiers sauvegardés.
- Bouton Streamlit pour déclencher la sauvegarde et éviter les doublons.
"""


def save_uploaded_files(uploaded_files, decks_dir, translated_dir):
    """
    Sauvegarde tous les fichiers uploadés et génère leurs fichiers TXT traduits.

    Étapes principales :
    1. Vérifie que des fichiers ont été uploadés, sinon affiche un avertissement.
    2. Vérifie si les fichiers ont déjà été sauvegardés dans `st.session_state`.
       a. Si oui, informe l'utilisateur et ne fait rien.
    3. Si le bouton Streamlit de sauvegarde est cliqué :
       a. Pour chaque fichier uploadé :
           i. Détermine le nom final (possiblement renommé par l'utilisateur).
           ii. Sauvegarde le fichier PDF dans `decks_dir`.
           iii. Extrait le texte du PDF et le traduit.
           iv. Sauvegarde le texte traduit dans `translated_dir` en fichier TXT.
           v. Ajoute le nom du fichier sauvegardé à la liste.
    4. Met à jour `st.session_state` :
       - `saved_uploaded_files` = True
       - `uploaded_files_saved_names` = liste des fichiers sauvegardés
    5. Affiche un message de succès et relance l'interface Streamlit.

    Paramètres
    ----------
    uploaded_files : list
        Liste des fichiers PDF uploadés via Streamlit.
    decks_dir : str
        Chemin du répertoire où sauvegarder les fichiers PDF.
    translated_dir : str
        Chemin du répertoire où sauvegarder les fichiers TXT traduits.

    Effets
    -------
    - Sauvegarde les fichiers PDF et TXT traduits.
    - Met à jour `st.session_state` avec les fichiers sauvegardés.
    - Affiche des messages Streamlit d'information, warning ou succès.
    """
    
    if not uploaded_files:
        st.warning("Aucun fichier à sauvegarder.")
        return

    if st.session_state.get("saved_uploaded_files", False):
        st.info("✅ Les fichiers ont déjà été sauvegardés.")
        return

    saved_files_names = st.session_state.get("uploaded_files_saved_names", [])
    if not isinstance(saved_files_names, list):
        saved_files_names = []

    if st.button("💾 Sauvegarder tous les fichiers uploadés"):
        for file in uploaded_files:
            original_name = file.name
            rename_key = f"rename_{original_name}"
            final_name = str(st.session_state.get(rename_key, original_name))

            # --- Sauvegarde du fichier PDF ---
            save_path = os.path.join(decks_dir, final_name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            # --- Extraction et traduction du texte, puis sauvegarde en TXT ---
            txt_path = os.path.join(translated_dir, os.path.splitext(final_name)[0] + ".txt")
            file.seek(0) # Revenir au début du fichier PDF
            uploaded_text = translate_text(extract_text_from_pdf(file))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(uploaded_text)
        

            # --- Ajout du nom du fichier sauvegardé à la liste ---
            saved_files_names.append(final_name)


            commit_file_to_github(
                local_path=save_path,
                repo_path=f"src/data/decks/{final_name}",
                commit_message=f"Ajout du deck {final_name} (PDF)"
            )
            print(f"🚀 {final_name} (PDF) commité sur GitHub avec succès !")

            commit_file_to_github(
                local_path=txt_path,
                repo_path=f"src/data/processed/translated/{os.path.basename(txt_path)}",
                commit_message=f"Ajout du texte traduit pour {final_name}"
            )
            print(f"🚀 {os.path.basename(txt_path)} (TXT) commité sur GitHub avec succès !")
            

        # --- Mise à jour de st.session_state après sauvegarde ---
        st.session_state.saved_uploaded_files = True
        st.session_state.uploaded_files_saved_names = saved_files_names

        st.success("✅ Tous les fichiers uploadés et leurs TXT traduits ont été sauvegardés.")
        st.rerun()
