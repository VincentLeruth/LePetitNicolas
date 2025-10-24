import streamlit as st

"""
Module Streamlit pour l'upload de fichiers PDF (decks) à analyser.

Fonctionnalités :
- Permet à l'utilisateur de sélectionner un ou plusieurs fichiers PDF.
- Filtrage : seuls les fichiers PDF avec noms en minuscules sont acceptés.
- Affichage des fichiers uploadés avec option de suppression.
- Gestion de l'état via `st.session_state` pour conserver les fichiers entre les reruns.
- Empêche la suppression si les fichiers ont déjà été sauvegardés.
"""


def upload_decks():
    """
    Interface Streamlit pour uploader des fichiers PDF à analyser.

    Étapes principales :
    1. Initialise `st.session_state.uploaded_files` si non existant.
    2. Affiche un file uploader seulement si aucun fichier n'est encore uploadé.
    3. Filtrage des fichiers uploadés :
       a. Valides : noms en minuscules et extension .pdf.
       b. Invalides : autres fichiers.
    4. Met à jour `st.session_state.uploaded_files` avec les fichiers valides.
    5. Affiche les fichiers uploadés avec option de suppression si les fichiers
       n'ont pas encore été sauvegardés.
    6. Supprime les fichiers de la session et toutes leurs clés associées si l'utilisateur choisit "Supprimer".
    
    Returns
    -------
    list
        Liste des fichiers PDF valides uploadés pour cette session.
    """
    
    st.subheader("📁 Sélection des fichiers PDF à analyser")

    # --- Initialisation de la session pour les fichiers uploadés ---
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # --- File uploader seulement si aucun fichier uploadé ---
    if not st.session_state.uploaded_files:
        uploaded_files = st.file_uploader(
            "Choisissez un ou plusieurs fichiers PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        if uploaded_files:
            valid_files = []
            invalid_files = []
            # --- Vérification des fichiers uploadés ---
            for file in uploaded_files:
                if file.name.islower() and file.name.endswith(".pdf"):
                    valid_files.append(file)
                else:
                    invalid_files.append(file.name)

            # --- Mise à jour des fichiers valides dans la session ---
            if valid_files:
                st.session_state.uploaded_files = valid_files
                st.success(f"{len(valid_files)} fichier(s) valide(s) chargé(s) ✅")

            # --- Affichage des fichiers invalides ---
            if invalid_files:
                st.error(f"{len(invalid_files)} fichier(s) rejeté(s) ❌")
                for name in invalid_files:
                    st.write(f"- {name}")

    # --- Affichage des fichiers uploadés avec option de suppression ---
    if st.session_state.uploaded_files:
        st.info("Fichiers uploadés pour cette session :")

        # Vérifie si les fichiers peuvent être supprimés
        can_remove = not st.session_state.get("saved_uploaded_files", False)

        remove_index = None
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4,1])
            with col1:
                st.write(f"- {file.name}")
            with col2:
                if can_remove and st.button("❌ Supprimer", key=f"remove_{file.name}"):
                    remove_index = i

        # --- Suppression du fichier et des clés associées ---
        if remove_index is not None:
            removed_file = st.session_state.uploaded_files.pop(remove_index)
            for key in list(st.session_state.keys()):
                if removed_file.name in key:
                    del st.session_state[key]
            # Pas besoin de rerun, la session actualisée suffit pour mettre à jour l'UI

    return st.session_state.uploaded_files
