"""
Module `predict_country` pour prédire le pays d'un document.

Ce module charge un modèle Gradient Boosting entraîné sur des vecteurs TF-IDF
afin de prédire la probabilité d'appartenance de chaque document à un pays donné.

Fonctionnalités principales :
- Chargement du modèle et de l'encodeur sauvegardés (LabelEncoder)
- Prédiction des probabilités par classe à partir de TF-IDF
- Application d’un seuil de confiance pour marquer les cas incertains en "unknown"
- Export d’un fichier CSV contenant :
    * les documents
    * la prédiction finale du pays
    * la probabilité associée
    * les probabilités détaillées pour chaque pays
"""

import os
import pandas as pd
import joblib
import numpy as np

from commite_github import commit_file_to_github


def predict_country():
    """
    Prédit le pays d’origine de chaque document à partir de ses vecteurs TF-IDF.

    Cette fonction charge un modèle Gradient Boosting et un encodeur de labels
    pour prédire la probabilité d’appartenance de chaque document à une classe
    de pays. Les résultats sont sauvegardés dans un fichier CSV de sortie avec
    toutes les probabilités et la classe prédite.

    Étapes principales :
    --------------------
    1. Charger le modèle et l’encodeur depuis le dossier `models/`.
    2. Charger les vecteurs TF-IDF depuis `data/processed/tfidf_vectors.csv`.
    3. Vérifier que les features correspondent à celles attendues par le modèle.
    4. Calculer les probabilités pour chaque classe (pays).
    5. Appliquer un seuil de confiance (`threshold`) pour filtrer les prédictions incertaines.
    6. Sauvegarder un CSV contenant toutes les prédictions et probabilités.

    Paramètres :
    ------------
    threshold : float, optionnel
        Seuil minimal de probabilité pour accepter une prédiction.
        Si la probabilité maximale < threshold, le pays est marqué "unknown".
        (Valeur par défaut : 0.3)
    """

    # --- Définition des chemins des fichiers nécessaires ---
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "..", "..", "models", "country_gb_model.joblib")
    le_path = os.path.join(base_dir, "..", "..", "..", "models", "country_label_encoder.joblib")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")

    # Dossier et fichier de sortie
    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tfidf_vectors_with_country_predictions.csv")
    
    # Seuil de probabilité minimale pour accepter une prédiction
    threshold = 0.3

    # --- Chargement du modèle Gradient Boosting et de l'encodeur de labels ---
    model = joblib.load(model_path)
    le = joblib.load(le_path)

    # --- Chargement des vecteurs TF-IDF ---
    df_vectors_full = pd.read_csv(vectors_path, sep=";")

    # --- Préparation des features pour la prédiction ---
    X = df_vectors_full.drop(columns=["doc"], errors="ignore")

    # Vérification que toutes les colonnes attendues par le modèle sont présentes
    feature_names = model.feature_names_in_
    for feat in feature_names:
        if feat not in X.columns:
            # Si une feature manque, on la crée avec des zéros
            X[feat] = 0

    # Réordonne les colonnes pour correspondre à l’ordre d’entraînement du modèle
    X = X[feature_names]

    # --- Prédiction des probabilités par classe ---
    proba_all = model.predict_proba(X)

    # Extraction de la meilleure probabilité et de la classe correspondante
    best_proba = np.max(proba_all, axis=1)
    best_class = np.argmax(proba_all, axis=1)

    # --- Application du seuil de confiance ---
    predictions = [
        le.inverse_transform([cls_idx])[0] if prob >= threshold else "unknown"
        for prob, cls_idx in zip(best_proba, best_class)
    ]

    # --- Construction du DataFrame des résultats ---
    # Si la colonne "doc" existe, on la garde, sinon on crée un identifiant numérique
    df_results = (
        df_vectors_full[["doc"]].copy()
        if "doc" in df_vectors_full.columns
        else pd.DataFrame({"doc": range(len(predictions))})
    )
    df_results["predicted_country"] = predictions
    df_results["confidence"] = best_proba

    # Ajout des probabilités complètes pour chaque pays (utile pour debug et analyse)
    for idx, class_name in enumerate(le.classes_):
        df_results[f"proba_{class_name}"] = proba_all[:, idx]

    # --- Sauvegarde des résultats github ---
    commit_file_to_github(
        local_file_path=output_file,  
        repo_path=output_file,        
        commit_message="Update country prediction results"
    )
    print("🚀 Résultats pays committés sur GitHub avec succès !")


# --- Point d’entrée principal ---
if __name__ == "__main__":
    predict_country()
