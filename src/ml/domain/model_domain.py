"""
Module `train_domain_select.py`

Ce script entraîne un modèle de classification supervisée pour prédire le *domaine*
associé à un document à partir de ses vecteurs TF-IDF.  
Il combine plusieurs étapes de sélection et réduction de dimensions, puis entraîne
un classifieur logistique avec pondération des classes.

Fonctionnalités principales :
-----------------------------
- Sélection de features via test du chi² (`SelectKBest`)
- Réduction dimensionnelle via SVD (`TruncatedSVD`)
- Calcul de similarités cosinus entre documents et centroïdes de classes
- Normalisation des features finales
- Entraînement d’un modèle de régression logistique pondérée
- Sauvegarde de tous les objets nécessaires à l’inférence : sélecteur, SVD, scaler,
  centroïdes, classifieur, encodeur et liste des colonnes non nulles

Fichiers d’entrée :
-------------------
- `data/processed/tfidf_vectors.csv` : vecteurs TF-IDF
- `data/labeled.csv` : labels de domaines

Fichiers de sortie :
--------------------
- `models/domain_selector.joblib` : sélecteur de features chi²
- `models/domain_svd.joblib` : modèle de réduction dimensionnelle SVD
- `models/domain_scaler.joblib` : scaler pour standardisation
- `models/domain_centroids.joblib` : centroïdes calculés pour chaque classe
- `models/domain_clf.joblib` : modèle de régression logistique entraîné
- `models/domain_label_encoder.joblib` : encodeur de labels
- `models/domain_nonzero_columns.joblib` : liste des colonnes non nulles utilisées
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from commite_github import commit_file_to_github

# --- Définition des chemins de base ---
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LAB_CSV  = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODELS_DIR = os.path.join(BASE, "..", "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Hyperparamètres globaux ---
K_SELECT = 3000        # Nombre de features à garder après sélection chi²
SVD_COMPONENTS = 150   # Nombre de composantes SVD (réduction dimensionnelle)
CLASS_WEIGHTED = True  # Active la pondération automatique des classes pour LogisticRegression


def _canon_label(s):
    """
    Normalise une étiquette textuelle (label) en la convertissant en minuscule et en retirant les espaces.

    Paramètres
    ----------
    s : str ou NaN
        Valeur brute du label.

    Retour
    ------
    str
        Label nettoyé en minuscules, ou "unknown" si la valeur est manquante.
    """
    return str(s).strip().lower() if pd.notna(s) else "unknown"


def train_domain():
    """
    Entraîne un modèle de classification du domaine à partir de vecteurs TF-IDF.

    Étapes principales :
    --------------------
    1. Chargement et fusion des données de vecteurs et labels.
    2. Nettoyage, conversion et suppression des features nulles.
    3. Sélection de K meilleures features via test du chi².
    4. Calcul de centroïdes moyens par classe et de similarités cosinus.
    5. Réduction dimensionnelle via SVD.
    6. Normalisation des features et concaténation des similarités.
    7. Entraînement du modèle de régression logistique (avec `class_weight='balanced'`).
    8. Sauvegarde de tous les objets nécessaires à la prédiction future.

    Exceptions
    ----------
    RuntimeError
        Si aucune correspondance entre vecteurs et labels n’est trouvée.
    """

    # --- Chargement des données ---
    df_vec = pd.read_csv(VECT_CSV, sep=";")
    df_lab = pd.read_csv(LAB_CSV, sep=";")

    # Conversion explicite des identifiants en chaîne
    df_vec["doc"] = df_vec["doc"].astype(str)
    df_lab["doc"] = df_lab["doc"].astype(str)

    # --- Fusion des données et préparation des labels ---
    df = pd.merge(df_vec, df_lab, on="doc", how="inner")
    if df.shape[0] == 0:
        raise RuntimeError("Aucune correspondance entre vecteurs et labels")
    
    # Canonisation des labels de domaine
    df["domain_y"] = df["domain_y"].apply(_canon_label)

    # --- Extraction des features ---
    X_df = df.drop(columns=["doc", "domain_y"], errors="ignore")

    # Conversion en numérique et remplacement des NaN
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Suppression des colonnes totalement nulles
    col_sums = X_df.sum(axis=0)
    nonzero_cols = col_sums[col_sums > 0].index.tolist()
    X_df = X_df[nonzero_cols]
    print("Features after zero-drop:", X_df.shape[1])

    # --- Préparation des labels ---
    y_raw = df["domain_y"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # --- Conversion en matrice sparse ---
    X_sp = csr_matrix(X_df.values)

    # --- Sélection de features via chi² ---
    k = min(K_SELECT, X_sp.shape[1] - 1)
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_sp, y)
    X_sel = selector.transform(X_sp)

    # --- Calcul des centroïdes par classe ---
    classes = np.unique(y)
    centroids = []
    for c in classes:
        rows = (y == c)
        if rows.sum() == 0:
            # Classe vide : vecteur nul
            centroids.append(np.zeros(X_sel.shape[1], dtype=float))
        else:
            s = X_sel[rows].sum(axis=0)
            centroid = np.asarray(s).ravel() / float(max(1, rows.sum()))
            centroids.append(centroid)
    centroids = np.vstack(centroids)

    # --- Similarité cosinus entre chaque document et les centroïdes ---
    cos_sim = cosine_similarity(X_sel, centroids)

    # --- Réduction dimensionnelle via SVD ---
    n_comp = min(SVD_COMPONENTS, X_sel.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X_sel)

    # --- Standardisation des features réduites ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_red)

    # --- Combinaison des features finales (SVD + similarités cosinus) ---
    X_final = np.hstack([X_scaled, cos_sim])

    # --- Entraînement du classifieur ---
    clf = LogisticRegression(
        class_weight="balanced" if CLASS_WEIGHTED else None,
        solver="saga",
        max_iter=2000
    )
    clf.fit(X_final, y)

    # --- Sauvegarde de tous les objets nécessaires à la prédiction ---
    files_to_commit = [
    os.path.join(MODELS_DIR, "domain_selector.joblib"),
    os.path.join(MODELS_DIR, "domain_svd.joblib"),
    os.path.join(MODELS_DIR, "domain_scaler.joblib"),
    os.path.join(MODELS_DIR, "domain_centroids.joblib"),
    os.path.join(MODELS_DIR, "domain_clf.joblib"),
    os.path.join(MODELS_DIR, "domain_label_encoder.joblib"),
    os.path.join(MODELS_DIR, "domain_nonzero_columns.joblib")
    ]
    
    for file_path in files_to_commit:
        commit_file_to_github(
            local_file_path=file_path,
            repo_path=file_path,  # conserve le même chemin dans le repo GitHub
            commit_message=f"Update {os.path.basename(file_path)}"
        )
        print(f"🚀 {os.path.basename(file_path)} commité sur GitHub avec succès !")


# --- Point d’entrée principal ---
if __name__ == "__main__":
    train_domain()
