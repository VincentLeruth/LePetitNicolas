import subprocess
import os
import streamlit as st
from urllib.parse import quote

GITHUB_USER = "Nic0o00"
GITHUB_REPO = "streamlit"

def sync_repo(repo_path, push=False, pull=False):
    """
    Synchronise le repo GitHub via la ligne de commande Git.
    Authentification HTTPS via token GitHub et username.
    Affiche un message 'Synchronisation en cours...' dans Streamlit.
    """
    token = os.environ.get("GITHUB_TOKEN")
    
    if not token:
        st.warning("⚠️ Aucun token GitHub trouvé dans les variables d'environnement.")
        return
    
    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        try:
            # URL HTTPS complète avec token
            url_cmd = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
            
            # Détecter la branche actuelle
            branch_result = subprocess.run(
                ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            branch = branch_result.stdout.strip()
            
            # Pull si demandé
            if pull:
                subprocess.run(
                    ["git", "-C", repo_path, "pull", url_cmd, branch],
                    check=True
                )
            
            # Push si demandé
            if push:
                # Ajouter tous les fichiers
                subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
                
                # Commit si nécessaire
                subprocess.run(
                    ["git", "-C", repo_path, "commit", "-m", "📤 Upload automatique depuis Streamlit"],
                    check=False  # échoue silencieusement si rien à commit
                )
                
                # Push vers GitHub
                subprocess.run(["git", "-C", repo_path, "push", url_cmd, branch], check=True)
            
            st.success("✅ Synchronisation terminée !")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Erreur Git : {e}")
