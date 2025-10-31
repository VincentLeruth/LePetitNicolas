import subprocess
import os
import streamlit as st
from urllib.parse import quote

GITHUB_USER = "Nic0o00"
GITHUB_REPO = "streamlit"

def run_cmd(cmd, repo_path, check=True, capture_output=False, text=True):
    return subprocess.run(cmd, cwd=repo_path, check=check,
                          capture_output=capture_output, text=text)

def has_unstaged_changes(repo_path):
    """Retourne True s'il y a des changements non-committés (tracked/modified/new)."""
    res = run_cmd(["git", "status", "--porcelain"], repo_path, check=True, capture_output=True)
    return bool(res.stdout.strip())

def has_commits_to_push(repo_path, branch):
    """
    Retourne True s'il y a des commits locaux qui ne sont pas encore sur le remote.
    Méthode robuste : on tente d'interroger l'upstream, si absent on considère qu'il faut pousser.
    """
    try:
        # S'assurer d'avoir la référence upstream
        # git rev-parse --abbrev-ref --symbolic-full-name @{u}
        run_cmd(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo_path, check=True)
    except subprocess.CalledProcessError:
        # Pas d'upstream configuré → on doit pousser (ou on laisse git push gérer)
        return True

    # Compare remote et local
    try:
        res = run_cmd(["git", "rev-list", "--left-right", "--count", f"@{{u}}...{branch}"], repo_path, check=True, capture_output=True)
        left_right = res.stdout.strip().split()
        if len(left_right) == 2:
            behind, ahead = map(int, left_right)
            # 'ahead' > 0 signifie que local a des commits que remote n'a pas
            return ahead > 0
    except subprocess.CalledProcessError:
        # si erreur, être conservateur : indiquer qu'il y a possiblement à pousser
        return True

    return False

def get_current_branch(repo_path):
    res = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path, check=True, capture_output=True)
    return res.stdout.strip()

def sync_repo(repo_path, push=False):
    """
    Synchronise le dépôt. 
    - Si push=False : fait un git pull pour récupérer les changements distants.
    - Si push=True  : ne pousse que s'il y a réellement des commits/changes à pousser.
    Cette fonction est idempotente (appel multiple ne provoque pas de commits/pushs vides).
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        st.warning("⚠️ Aucun token GitHub trouvé dans les variables d'environnement.")
        return

    with st.spinner("🔄 Synchronisation en cours avec GitHub..."):
        try:
            branch = get_current_branch(repo_path)
            url_cmd = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

            # Config temporaire Git (si nécessaire)
            run_cmd(["git", "config", "--local", "user.name", "Streamlit Bot"], repo_path)
            run_cmd(["git", "config", "--local", "user.email", "bot@localhost"], repo_path)

            if not push:
                # Pull : simple, tolérant aux conflits — laisse Git gérer les erreurs
                run_cmd(["git", "pull", url_cmd, branch], repo_path)
                st.success("✅ Pull terminé.")
                return

            # push == True : vérifier d'abord s'il y a quelque chose à pousser
            # 1) si workspace a des changements, on les ajoute et on commit
            if has_unstaged_changes(repo_path):
                # Stage all changed files
                run_cmd(["git", "add", "."], repo_path)
                # Commit (ne fait rien si rien à commit)
                # On capture la sortie pour détecter "nothing to commit"
                try:
                    run_cmd(["git", "commit", "-m", "📤 Upload automatique depuis Streamlit"], repo_path, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    # git commit renvoie une erreur (par ex. "nothing to commit"), on ignore
                    pass

            # 2) vérifier s'il y a des commits locaux à pousser
            if has_commits_to_push(repo_path, branch):
                run_cmd(["git", "push", url_cmd, branch], repo_path)
                st.success("✅ Push terminé.")
            else:
                st.info("ℹ️ Rien à pousser (repo à jour).")

        except subprocess.CalledProcessError as e:
            # Affiche l'erreur technique brute (utile au debug), sans casser l'app
            st.error(f"❌ Erreur Git : {e}")
