# utils/sync_repo.py
import streamlit as st
import git
import os
import time

def sync_repo(repo_path, push=False):
    """
    Synchronise le repo GitHub : pull automatique, push optionnel.
    Affiche un message 'Synchronisation en cours...' dans Streamlit.

    Parameters
    ----------
    repo_path : str
        Chemin local vers le repo cloné.
    push : bool
        Si True, fait un push des modifications vers GitHub (nécessite GITHUB_TOKEN).
    """
    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        try:
            repo = git.Repo(repo_path)
            
            # Pull
            repo.remotes.origin.pull()
            
            # Push si demandé
            if push:
                token = os.environ.get("GITHUB_TOKEN")
                if not token:
                    st.warning("⚠️ Aucun token GitHub trouvé, push ignoré.")
                else:
                    url = repo.remotes.origin.url
                    if url.startswith("https://"):
                        url_with_token = url.replace(
                            "https://", f"https://{token}@"
                        )
                        repo.remotes.origin.set_url(url_with_token)
                    
                    repo.git.add(all=True)
                    repo.index.commit("📤 Upload automatique depuis Streamlit")
                    repo.remotes.origin.push()
                    
                    repo.remotes.origin.set_url(url)

            time.sleep(1)
            st.success("✅ Synchronisation terminée !")
        except Exception as e:
            st.error(f"❌ Erreur lors de la synchronisation : {e}")
