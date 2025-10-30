"""
Script d'entraînement d'un modèle Random Forest pour classifier les documents selon leur résultat.

Entrées :
    - data/processed/tfidf_vectors.csv : vecteurs TF-IDF avec identifiant 'doc'
    - data/labeled.csv : fichier contenant les labels ('result') associés aux documents

Sortie :
    - models/deck_classifier_rf.joblib : modèle RandomForest entraîné
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Définition des chemins de base ---
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier_rf.joblib")

def train_result():
    """
    Fonction principale d'entraînement du modèle RandomForestClassifier.
    """

    # --- Charger vecteurs et labels ---
    X = pd.read_csv(VECT_CSV, sep=";")
    df_labels = pd.read_csv(LABELED_CSV, sep=";", encoding="ISO-8859-1")

    # --- Nettoyer les labels ---
    allowed_labels = ["Interesting", "Unfavorable", "Very Unfavorable", "Out"]
    df_labels = df_labels[df_labels["result"].isin(allowed_labels)]

    # --- Merge pour aligner vecteurs et labels ---
    df_merged = pd.merge(
        X, df_labels[["doc", "result"]],
        on="doc",
        how="inner"
    )

    # --- Préparer les données d'entrée pour l'entraînement ---
    X_train_vectors = df_merged.drop(columns=["doc", "result"])
    y_train_labels = df_merged["result"]

    # --- Vérification de l'alignement ---
    print(f"X shape: {X_train_vectors.shape}, y length: {len(y_train_labels)}")

    # --- Entraîner le modèle RandomForest ---
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_vectors, y_train_labels)

    # --- Évaluation rapide sur les données d'entraînement ---
    y_pred = clf.predict(X_train_vectors)
    print("=== Évaluation sur les données d'entraînement ===")
    print(classification_report(y_train_labels, y_pred))

    # --- Sauvegarde du modèle entraîné ---
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans {MODEL_PATH}")


# --- Point d’entrée du script ---
if __name__ == "__main__":
    train_result()
