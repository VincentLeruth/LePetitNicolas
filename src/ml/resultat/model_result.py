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
    print("Chargement des données...")
    X = pd.read_csv(VECT_CSV, sep=";", encoding="utf-8")
    df_labels = pd.read_csv(LABELED_CSV, sep=";", encoding="utf-8")

    # --- Nettoyage des colonnes critiques ---
    for df in [X, df_labels]:
        df["doc"] = df["doc"].astype(str).str.strip().str.lower()

    df_labels["resultat"] = df_labels["resultat"].astype(str).str.strip()

    # --- Filtrage des labels autorisés ---
    allowed_labels = ["Interesting", "Unfavorable", "Very Unfavorable", "Out"]
    df_labels = df_labels[df_labels["resultat"].isin(allowed_labels)]
    

    # --- Diagnostic sur les correspondances de documents ---
    docs_X = set(X["doc"])
    docs_labels = set(df_labels["doc"])

    missing_in_labels = docs_X - docs_labels
    missing_in_vectors = docs_labels - docs_X

    print(f"\nDiagnostic correspondances :")
    print(f" - {len(missing_in_labels)} documents TF-IDF sans label")
    print(f" - {len(missing_in_vectors)} labels sans vecteurs TF-IDF\n")

    # --- Fusion pour aligner vecteurs et labels ---
    df_merged = pd.merge(
        X,
        df_labels[["doc", "resultat"]],
        on="doc",
        how="inner"  # conserver uniquement les docs communs
    )
    print(df_merged.columns)
    

    if df_merged.empty:
        raise ValueError("Aucun document commun entre tfidf_vectors.csv et labeled.csv.")

    print(f"Fusion réussie : {len(df_merged)} documents alignés.")
    print(f"Colonnes disponibles après merge : {df_merged.columns.tolist()}")

    # --- Préparation des données d'entraînement ---
    X_train_vectors = df_merged.drop(columns=["doc", "resultat"])
    y_train_labels = df_merged["resultat"]

    print(f"\nDonnées prêtes : X shape = {X_train_vectors.shape}, y length = {len(y_train_labels)}")

    # --- Entraînement du modèle RandomForest ---
    print("\nEntraînement du modèle Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_vectors, y_train_labels)

    # --- Évaluation sur les données d'entraînement ---
    print("\n=== Évaluation sur les données d'entraînement ===")
    y_pred = clf.predict(X_train_vectors)
    print(classification_report(y_train_labels, y_pred))

    # --- Sauvegarde du modèle ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")


# --- Point d’entrée du script ---
if __name__ == "__main__":
    train_result()
