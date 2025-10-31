import subprocess
import os
import streamlit as st
from urllib.parse import quote

# ⚙️ À modifier selon ton compte GitHub
GITHUB_USER = "Nic0o00"
GITHUB_REPO = "streamlit"

def sync_repo(repo_path, push=False):
    """
    Synchronise le repo GitHub via la ligne de commande Git.
    Authentification HTTPS via token GitHub et username.
    Configure temporairement user.name et user.email pour permettre les commits automatiques.
    """
    token = os.environ.get("GITHUB_TOKEN")
    
    if not token:
        st.warning("⚠️ Aucun token GitHub trouvé dans les variables d'environnement.")
        return
    
    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        # Construire l'URL HTTPS complète avec token
        url_cmd = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
        
        # Détecter la branche actuelle
        branch_result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        branch = branch_result.stdout.strip()

        # ⚙️ Configurer temporairement l'identité Git pour ce dépôt
        subprocess.run(["git", "-C", repo_path, "config", "user.name", "Streamlit Bot"], check=True)
        subprocess.run(["git", "-C", repo_path, "config", "user.email", "bot@localhost"], check=True)
        
        # Pull si demandé
        if push == False:
            subprocess.run(
                ["git", "-C", repo_path, "pull", url_cmd, branch],
                check=True
            )
        
        # Push si demandé
        if push == True:
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