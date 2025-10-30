import streamlit as st
import git
import os
import time

def sync_repo(repo_path, push=False):
    """
    Synchronise le repo GitHub : pull automatique, push optionnel.
    Authentification HTTPS via token GitHub.
    Affiche un message 'Synchronisation en cours...' dans Streamlit.

    Parameters
    ----------
    repo_path : str
        Chemin local vers le repo cloné.
    push : bool
        Si True, fait un push des modifications vers GitHub.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        st.warning("⚠️ Aucun token GitHub trouvé dans les variables d'environnement.")
        return

    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        try:
            repo = git.Repo(repo_path)
            origin = repo.remotes.origin
            original_url = origin.url

            # Modifier l'URL pour inclure le token
            if original_url.startswith("https://"):
                url_with_token = original_url.replace("https://", f"https://{token}@")
                origin.set_url(url_with_token)

            # Pull des dernières modifications
            origin.pull()

            # Push si demandé
            if push:
                repo.git.add(all=True)
                repo.index.commit("📤 Upload automatique depuis Streamlit")
                origin.push()

            # Rétablir l'URL originale
            origin.set_url(original_url)

            time.sleep(1)  # pause pour le spinner
            st.success("✅ Synchronisation terminée !")
        except Exception as e:
            st.error(f"❌ Erreur lors de la synchronisation : {e}")
