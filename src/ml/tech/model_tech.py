"""
Module : train_tech
====================

Ce module entraîne un classifieur **multi-label** pour la technologie d’un document 
(en distinguant les catégories "hard" et "soft") à partir des **vecteurs TF-IDF enrichis** (1–3 n-grams).

Le modèle utilisé est une **régression logistique One-vs-Rest**, adaptée au multi-label :
un document peut appartenir à la fois aux deux classes.

Fonction principale :
---------------------
- `train_tech()` : entraîne, évalue, puis sauvegarde le modèle final.

Entrées :
---------
- `data/processed/tfidf_vectors.csv` : fichier contenant les vecteurs TF-IDF avec identifiant `doc`
- `data/labeled.csv` : fichier contenant les labels associés (`tech_y`)

Sorties :
---------
- `models/lr_multilabel_techno_model.joblib` : modèle entraîné
- Affichage console des rapports de classification (précision, rappel, F1-score)

Dépendances :
-------------
- pandas
- scikit-learn
- joblib
- os

Auteur :
--------
Fait partie de la pipeline de classification documentaire par TF-IDF enrichi.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from commite_github import commit_file_to_github


def train_tech():
    """
    Entraîne un modèle **multi-label Logistic Regression** pour prédire les technologies
    (`hard`, `soft`, ou `both`) à partir des vecteurs TF-IDF enrichis.

    Étapes principales :
    --------------------
    1. Chargement des vecteurs TF-IDF et des labels (`labeled.csv`)
    2. Nettoyage, harmonisation et fusion des données
    3. Encodage des labels multi-label (`hard`, `soft`)
    4. Séparation train/test (avec stratification)
    5. Entraînement du modèle One-vs-Rest Logistic Regression
    6. Évaluation du modèle sur le jeu de test
    7. Ré-entraînement sur tout le dataset et sauvegarde du modèle final

    Returns
    -------
    None
        Le modèle est sauvegardé dans le répertoire `models/` sous le nom
        `lr_multilabel_techno_model.joblib`.
    """
    # --- Chemins relatifs ---
    base_dir = os.path.dirname(__file__)
    path_vectors = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    path_labels  = os.path.join(base_dir, "..", "..", "..", "data", "labeled.csv")
    models_dir   = os.path.join(base_dir, "..", "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Chargement des données ---
    df_vec = pd.read_csv(path_vectors, sep=";")
    df_lab = pd.read_csv(path_labels, sep=";")

    # --- Harmonisation des types ---
    df_vec["doc"] = df_vec["doc"].astype(str)
    df_lab["doc"] = df_lab["doc"].astype(str)
    df_lab = df_lab[["doc", "tech"]]  # ne garde que les colonnes pertinentes

    # --- Uniformisation des noms de documents ---
    df_vec["doc"] = df_vec["doc"].str.replace(r"\.txt$", "", regex=True)

    # --- Fusion vecteurs + labels ---
    df = pd.merge(df_vec, df_lab, on="doc", how="inner")
    print(f"Lignes après merge : {len(df)}")
    if len(df) == 0:
        raise ValueError("Aucune correspondance entre vos vecteurs et vos labels. Vérifiez les noms de doc.")

    # --- Suppression des colonnes non utiles ---
    drop_cols = ["domain", "country", "client_y", "revenu", "gotomarket", "startup", "produit"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- Nettoyage des labels manquants ---
    df = df.dropna(subset=["tech_y"])
    if len(df) == 0:
        raise ValueError("Aucune ligne avec label valide après nettoyage.")

    # --- Création des labels binaires pour multi-label ---
    df["hard"] = df["tech_y"].apply(lambda x: 1 if x in ["hard", "both"] else 0)
    df["soft"] = df["tech_y"].apply(lambda x: 1 if x in ["soft", "both"] else 0)

    # --- Séparation features / cibles ---
    X = df.drop(columns=["doc", "tech_y", "hard", "soft"])
    y = df[["hard", "soft"]]

    print(f"Nombre de lignes après nettoyage : {len(df)}")
    print(df[["doc", "tech_y", "hard", "soft"]].head())

    # --- Stratification multi-label (pour équilibre) ---
    stratify_labels = df["hard"].astype(str) + df["soft"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify_labels
    )

    # --- Configuration du modèle ---
    base_clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=2.0,
        max_iter=2000,
        class_weight="balanced"
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)

    # --- Entraînement ---
    clf.fit(X_train, y_train)

    # --- Évaluation ---
    y_pred = clf.predict(X_test)
    print("\nRapport classification pour 'hard' :")
    print(classification_report(y_test["hard"], y_pred[:, 0], zero_division=0))
    print("\nRapport classification pour 'soft' :")
    print(classification_report(y_test["soft"], y_pred[:, 1], zero_division=0))

    # --- Réentraînement sur tout le dataset ---
    clf.fit(X, y)

    # --- Sauvegarde du modèle ---
    model_path = os.path.join(models_dir, "lr_multilabel_techno_model.joblib")
    commit_file_to_github(
        local_file_path=model_path,
        repo_path=model_path,  # même chemin dans le repo GitHub
        commit_message=f"Update {os.path.basename(model_path)}"
    )
    print(f"🚀 Modèle {os.path.basename(model_path)} commité sur GitHub avec succès !")


# --- Exécution directe ---
if __name__ == "__main__":
    train_tech()
