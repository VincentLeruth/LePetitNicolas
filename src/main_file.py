import os
from treatment.extract_text import extract_text_from_pdf
from treatment.detect_lang import detect_language
from treatment.translate import translate_text

from vectorisation.vectorize_text import vectorize_text

from ml.tech.predict_tech import predict_tech
from ml.domain.predict_domain import predict_domain
from ml.country.predict_country import predict_country
from ml.resultat.predict_resultat import predict_resultat

from ml.evaluate import evaluate

"""
Script de traitement complet des decks PDF.

Fonctionnalités :
- Parcourt tous les fichiers PDF dans le dossier 'data/decks'.
- Extraction du texte brut depuis les PDFs.
- Détection automatique de la langue et traduction en anglais si nécessaire.
- Sauvegarde du texte traduit dans 'data/processed/translated'.
- Vectorisation TF-IDF sur tous les documents.
- Prédiction pour tous les modèles : tech, domain, country, result.
- Évaluation des modèles sur chaque axe.
"""

# --- Chemins principaux ---
DECK_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "decks")
TRANSLATED_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "translated")

# Crée le dossier de textes traduits s'il n'existe pas
os.makedirs(TRANSLATED_DIRECTORY, exist_ok=True)

# --- Listes pour stocker noms de fichiers et textes ---
docs = []
texts = []

# --- Parcours des fichiers PDF ---
for each_file in os.listdir(DECK_DIRECTORY):
    deck_path = os.path.join(DECK_DIRECTORY, each_file)
    translated_path = os.path.join(TRANSLATED_DIRECTORY, each_file.replace(".pdf", ".txt"))

    # --- Chargement texte traduit si déjà existant ---
    if os.path.exists(translated_path):
        with open(translated_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # --- Extraction texte brut depuis PDF ---
        text = extract_text_from_pdf(deck_path)
        # --- Détection de la langue ---
        lang = detect_language(text)
        # --- Traduction si non anglais ---
        if lang != "en":
            text = translate_text(text)
        if text is None:
            text = "none"
        # --- Sauvegarde du texte traduit ---
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(text)

    # --- Ajout aux listes pour traitement ultérieur ---
    docs.append(each_file)      # Nom du fichier
    texts.append(text)          # Texte (traduit si nécessaire)

# --- Vectorisation TF-IDF globale ---
vectorize_text()

# --- Prédictions pour tous les axes ---
predict_tech()
predict_domain()
predict_country()
predict_resultat()

# --- Évaluation des modèles ---
evaluate("tech")
evaluate("domain")
evaluate("country")
evaluate("resultat")
