from github import Github, GithubException
import os
import streamlit as st

# --- Connexion à GitHub via token stocké dans Streamlit Secrets ---
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO_NAME = "streamlit"
BRANCH = "main"

g = Github(st.secrets["GITHUB_TOKEN"])
repo = g.get_repo(f"{st.secrets['GITHUB_USER']}/{st.secrets['GITHUB_REPO']}")



def commit_file_to_github(local_path, repo_path, commit_message):
    if not os.path.exists(local_path):
        st.warning(f"Le fichier {local_path} n'existe pas localement.")
        return
    
    with open(local_path, "rb") as f:
        content = f.read()

    try:
        existing_file = repo.get_contents(repo_path, ref=BRANCH)
        repo.update_file(existing_file.path, commit_message, content, existing_file.sha, branch=BRANCH)
        st.info(f"✅ Mise à jour sur GitHub : {repo_path}")
    except GithubException as e:
        if e.status == 404:
            # Le fichier n'existe pas → création
            repo.create_file(repo_path, commit_message, content, branch=BRANCH)
            st.info(f"✅ Création sur GitHub : {repo_path}")
        else:
            st.error(f"❌ Erreur GitHub : {e}")
