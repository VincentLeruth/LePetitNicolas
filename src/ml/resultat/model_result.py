"""
Script d'entraînement d'un modèle Random Forest pour classifier les documents selon leur résultat.

Ce module :
1. Charge les vecteurs TF-IDF et les labels de classification.
2. Filtre et nettoie les données pour ne conserver que les entrées valides.
3. Entraîne un modèle de classification RandomForest sur l’ensemble du jeu de données.
4. Évalue le modèle sur les données d’entraînement (évaluation interne rapide).
5. Sauvegarde le modèle entraîné pour une utilisation ultérieure.

Entrées :
    - data/processed/tfidf_vectors.csv : vecteurs TF-IDF avec identifiant 'doc'
    - data/labeled.csv : fichier contenant les labels ('resultat') associés aux documents

Sortie :
    - models/deck_classifier_rf.joblib : modèle RandomForest entraîné

Auteur :
    Script conçu pour un pipeline de classification documentaire basé sur des représentations TF-IDF.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from commite_github import commit_file_to_github

# --- Définition des chemins de base ---
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier_rf.joblib")

def train_result():
    """
    Fonction principale d'entraînement du modèle RandomForestClassifier.

    Étapes :
        1. Chargement des vecteurs TF-IDF et des labels.
        2. Nettoyage des labels pour ne conserver que les catégories valides.
        3. Alignement des jeux de données (vecteurs ↔ labels).
        4. Entraînement d’un modèle RandomForest sur les données complètes.
        5. Évaluation rapide sur les données d’entraînement.
        6. Sauvegarde du modèle entraîné au format .joblib.
    """
    # --- Charger vecteurs et labels ---
    # Lecture du fichier des vecteurs TF-IDF (avec colonne 'doc')
    X = pd.read_csv(VECT_CSV, sep=";")
    # Lecture du fichier des labels avec encodage ISO-8859-1 pour compatibilité
    df_labels = pd.read_csv(LABELED_CSV, sep=";", encoding="ISO-8859-1")

    # --- Nettoyer les labels ---
    # On garde uniquement les labels appartenant à la liste des valeurs autorisées
    allowed_labels = ["Interesting", "Unfavorable", "Very Unfavorable", "Out"]
    df_labels = df_labels[df_labels["result"].isin(allowed_labels)]

    # --- Filtrer les vecteurs ---
    # Ne conserver que les vecteurs correspondant à des documents labellisés
    X_train_vectors = X[X["doc"].isin(df_labels["doc"])].reset_index(drop=True)

    # --- Extraire les labels correspondants ---
    # Alignement entre les vecteurs et les labels sur la colonne 'doc'
    y_train_labels = df_labels[df_labels["doc"].isin(X_train_vectors["doc"])].reset_index(drop=True)["result"]

    # --- Préparer les données d'entrée pour l'entraînement ---
    # Suppression de la colonne 'doc' (non utilisée par le modèle)
    X_train_vectors = X_train_vectors.drop(columns=["doc"])

    # --- Entraîner le modèle RandomForest ---
    # Utilisation d’un modèle équilibré pour compenser les déséquilibres de classes
    clf = RandomForestClassifier(
        n_estimators=500,      # nombre d’arbres dans la forêt
        class_weight="balanced",  # pondération automatique des classes rares
        random_state=42,       # reproductibilité
        n_jobs=-1              # parallélisation complète
    )
    clf.fit(X_train_vectors, y_train_labels)

    # --- Évaluation rapide sur les données d'entraînement ---
    # (permet de vérifier le bon apprentissage, même si non représentatif du test réel)
    y_pred = clf.predict(X_train_vectors)
    print("=== Évaluation sur les données d'entraînement ===")
    print(classification_report(y_train_labels, y_pred))

    # --- Sauvegarde du modèle entraîné ---
    commit_file_to_github(
        local_file_path=MODEL_PATH,
        repo_path=MODEL_PATH,  # même chemin dans le repo GitHub
        commit_message=f"Update {os.path.basename(MODEL_PATH)}"
    )
    print(f"🚀 Modèle {os.path.basename(MODEL_PATH)} commité sur GitHub avec succès !")

# --- Point d’entrée du script ---
if __name__ == "__main__":
    train_result()
