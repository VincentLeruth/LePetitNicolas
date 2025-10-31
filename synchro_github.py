import subprocess
import os
import streamlit as st
from urllib.parse import quote

# ⚙️ Récupérer le user et le repo depuis les variables d'environnement ou secrets Streamlit
GITHUB_USER = os.environ.get("GITHUB_USER") or st.secrets.get("GITHUB_USER")
GITHUB_REPO = os.environ.get("GITHUB_REPO") or st.secrets.get("GITHUB_REPO")
TOKEN = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")

def sync_repo(repo_path, push=False):
    """
    Synchronise un dépôt Git local avec GitHub via la ligne de commande Git.
    Pull ou push automatique selon le paramètre `push`.
    """
    
    
    if not TOKEN:
        st.warning("⚠️ Aucun token GitHub trouvé dans les variables d'environnement ou secrets.")
        return
    
    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        url_cmd = f"https://{GITHUB_USER}:{quote(TOKEN)}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
        
        branch_result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        branch = branch_result.stdout.strip()

        subprocess.run(["git", "-C", repo_path, "config", "user.name", "Streamlit Bot"], check=True)
        subprocess.run(["git", "-C", repo_path, "config", "user.email", "bot@localhost"], check=True)
        
        if not push:
            subprocess.run(["git", "-C", repo_path, "pull", url_cmd, branch], check=True)
        else:
            subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
            subprocess.run(
                ["git", "-C", repo_path, "commit", "-m", "📤 Upload automatique depuis Streamlit"],
                check=False
            )
            subprocess.run(["git", "-C", repo_path, "push", url_cmd, branch], check=True)
        
        st.success("✅ Synchronisation terminée !")
