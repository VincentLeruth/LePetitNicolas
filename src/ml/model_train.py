import os
import pandas as pd

from ml.resultat.model_result import train_resultat
from src.ml.country.model_country import train_country
from src.ml.domain.model_domain import train_domain
from src.ml.tech.model_tech import train_tech

"""
Module utilitaire pour la gestion des decks labellisés et l'entraînement des modèles.

Fonctionnalités :
- Récupérer la liste des decks non labellisés.
- Ajouter ou mettre à jour les corrections dans labeled.csv.
- Lancer l'entraînement de tous les modèles ML existants (tech, domain, country, resultat).
"""

# --- Chemins principaux ---
BASE_DIR = os.path.dirname(__file__) 
DECK_DIR = os.path.join(BASE_DIR, "..", "..", "data", "decks") 
LABELED_CSV = os.path.join(BASE_DIR, "..","..", "..", "labeled.csv")


def get_unlabeled_decks():
    """
    Retourne la liste des fichiers PDF qui ne sont pas encore présents dans labeled.csv.

    Returns
    -------
    list of str
        Liste des noms de fichiers PDF non labellisés.
    """
    if os.path.exists(LABELED_CSV):
        labeled = pd.read_csv(LABELED_CSV, sep=";")
        labeled_files = labeled['doc'].tolist()
    else:
        labeled_files = []

    all_decks = [f for f in os.listdir(DECK_DIR) if f.endswith(".pdf")]
    unlabeled = [f for f in all_decks if f not in labeled_files]
    return unlabeled


def save_corrections(deck_name, corrections):
    """
    Ajoute ou met à jour les corrections pour un deck dans labeled.csv.

    Parameters
    ----------
    deck_name : str
        Nom du fichier PDF du deck.
    corrections : dict
        Dictionnaire contenant les corrections pour les colonnes :
        "tech", "domain", "country", "resultat".
    """
    if os.path.exists(LABELED_CSV):
        df = pd.read_csv(LABELED_CSV)
    else:
        df = pd.DataFrame(columns=["doc", "tech", "domain", "country", "resultat"])

    # Vérifie si le deck existe déjà et met à jour si nécessaire
    if deck_name in df["doc"].values:
        df.loc[df["doc"] == deck_name, ["doc", "tech", "domain", "country", "resultat"]] = [
            deck_name, corrections["tech"], corrections["domain"], corrections["country"], corrections["resultat"]
        ]
    else:
        row = {"doc": deck_name, **corrections}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(LABELED_CSV, index=False)


def train_all_models():
    """
    Lance l'entraînement de tous les modèles ML existants dans l'ordre :
    resultat, country, domain, tech.
    Affiche l'état de progression dans la console.
    """
    print("Entraînement du modèle 'resultat'...")
    train_resultat()

    print("Entraînement du modèle 'country'...")
    train_country()

    print("Entraînement du modèle 'domain'...")
    train_domain()

    print("Entraînement du modèle 'tech'...")
    train_tech()

    print("Tous les modèles ont été entraînés !")
