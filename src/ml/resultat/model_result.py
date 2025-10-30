"""
Module : predict_resultat
=========================

Ce module effectue la prédiction du résultat (label final) d’un document
à partir des vecteurs TF-IDF et d’un modèle de classification entraîné
(prévision type "success/failure" ou autre label final).

Fonctionnalités principales :
-----------------------------
- Chargement du modèle de classification (`lr_result_model.joblib`)
- Chargement des vecteurs TF-IDF depuis `data/processed/tfidf_vectors.csv`
- Vérification des features manquantes et remplissage avec 0
- Prédiction du label final pour chaque document
- Export du CSV contenant `doc`, `predicted_result`, et `confidence_score`

Fichier de sortie :
-------------------
- `output/predictions/tfidf_vectors_with_result_predictions.csv`
"""

import os
import pandas as pd
import joblib
import numpy as np


def predict_resultat():
    """Effectue les prédictions du résultat pour chaque document."""

    # --- Définition des chemins ---
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "..", "..", "models", "lr_result_model.joblib")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")

    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tfidf_vectors_with_result_predictions.csv")

    # --- Chargement du modèle entraîné ---
    clf = joblib.load(model_path)

    # --- Chargement des vecteurs TF-IDF ---
    df_vectors = pd.read_csv(vectors_path, sep=";")

    # --- Vérifie la présence de la colonne 'doc' ---
    if "doc" not in df_vectors.columns:
        df_vectors["doc"] = df_vectors.index.astype(str)

    # --- Vérification des features manquantes ---
    feature_names = clf.feature_names_in_
    missing_feats = [feat for feat in feature_names if feat not in df_vectors.columns]

    if missing_feats:
        zeros_df = pd.DataFrame(0, index=df_vectors.index, columns=missing_feats)
        df_vectors = pd.concat([df_vectors, zeros_df], axis=1)

    # --- Sélection des colonnes pour la prédiction ---
    X = df_vectors[feature_names]

    # --- Prédiction des labels ---
    preds = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        confidence = clf.predict_proba(X).max(axis=1)
    else:
        confidence = np.ones(len(X))

    # --- Construction du DataFrame final ---
    df_results = pd.DataFrame({
        "doc": df_vectors["doc"],
        "predicted_result": preds,
        "confidence_score": confidence
    })

    # --- Sauvegarde ---
    df_results.to_csv(output_file, index=False, sep=";")
    print(f"✅ Prédictions 'resultat' sauvegardées dans : {output_file}")


# --- Point d’entrée principal ---
if __name__ == "__main__":
    predict_resultat()
