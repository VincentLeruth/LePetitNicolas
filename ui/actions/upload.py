import streamlit as st

def upload_decks():
    st.subheader("📁 Sélection des fichiers PDF à analyser")

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
            for file in uploaded_files:
                if file.name.islower() and file.name.endswith(".pdf"):
                    valid_files.append(file)
                else:
                    invalid_files.append(file.name)

            if valid_files:
                st.session_state.uploaded_files = valid_files
                st.success(f"{len(valid_files)} fichier(s) valide(s) chargé(s) ✅")

            if invalid_files:
                st.error(f"{len(invalid_files)} fichier(s) rejeté(s) ❌")
                for name in invalid_files:
                    st.write(f"- {name}")

    # --- Afficher les fichiers uploadés avec option supprimer ---
    if st.session_state.uploaded_files:
        st.info("Fichiers uploadés pour cette session :")

        # Si les fichiers ont été sauvegardés, on ne peut plus les supprimer
        can_remove = not st.session_state.get("saved_uploaded_files", False)

        remove_index = None
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4,1])
            with col1:
                st.write(f"- {file.name}")
            with col2:
                if can_remove and st.button("❌ Supprimer", key=f"remove_{file.name}"):
                    remove_index = i

        if remove_index is not None:
            removed_file = st.session_state.uploaded_files.pop(remove_index)
            # Supprimer toutes les clés associées
            for key in list(st.session_state.keys()):
                if removed_file.name in key:
                    del st.session_state[key]
            # Pas besoin de rerun, l'état session actualisé suffit pour que l'UI se mette à jour

    return st.session_state.uploaded_files
