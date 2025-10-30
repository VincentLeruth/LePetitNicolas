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
    import os
    import streamlit as st
    from github import Github, GithubException

    # Vérifie que le fichier existe localement
    if not os.path.exists(local_path):
        st.warning(f"Le fichier {local_path} n'existe pas localement.")
        return
    
    # Lecture du fichier
    with open(local_path, "rb") as f:
        content = f.read()

    try:
        # Tente de récupérer le fichier dans GitHub
        existing_file = repo.get_contents(repo_path, ref=BRANCH)
        repo.update_file(existing_file.path, commit_message, content, existing_file.sha, branch=BRANCH)
        st.info(f"✅ Mise à jour sur GitHub : {repo_path}")
    except GithubException as e:
        if e.status == 404:
            # Si le fichier n'existe pas, création
            repo.create_file(repo_path, commit_message, content, branch=BRANCH)
            st.info(f"✅ Création sur GitHub : {repo_path}")
        else:
            st.error(f"❌ Erreur GitHub : {e}")

