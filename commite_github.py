from github import Github
import os
import streamlit as st

# --- Connexion à GitHub via token stocké dans Streamlit Secrets ---
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO_NAME = "https://github.com/Nic0o00/streamlit"
BRANCH = "main"

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

def commit_file_to_github(local_path, repo_path, commit_message):
    """
    Upload ou met à jour un fichier sur GitHub.
    
    Parameters
    ----------
    local_path : str
        Chemin local sur le serveur Streamlit
    repo_path : str
        Chemin relatif dans le repo GitHub
    commit_message : str
        Message du commit
    """
    if not os.path.exists(local_path):
        st.warning(f"Le fichier {local_path} n'existe pas localement.")
        return
    
    with open(local_path, "rb") as f:
        content = f.read()

    try:
        # Si le fichier existe déjà → mise à jour
        existing_file = repo.get_contents(repo_path, ref=BRANCH)
        repo.update_file(existing_file.path, commit_message, content, existing_file.sha, branch=BRANCH)
        st.info(f"✅ Mise à jour sur GitHub : {repo_path}")
    except:
        # Sinon → création
        repo.create_file(repo_path, commit_message, content, branch=BRANCH)
        st.info(f"✅ Création sur GitHub : {repo_path}")
