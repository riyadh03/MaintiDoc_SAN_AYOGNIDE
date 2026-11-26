"""Configuration centralisée du projet MaintiDoc."""
from pathlib import Path
import os

# Détection de la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Chemins des données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Document source par défaut
DEFAULT_PDF_PATH = RAW_DATA_DIR / "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"
if not DEFAULT_PDF_PATH.exists():
    # Fallback à la racine si pas encore déplacé
    DEFAULT_PDF_PATH = PROJECT_ROOT / "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"

# Dossiers d'extraction
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
TABLES_DIR = PROCESSED_DATA_DIR / "tables"
TEXTS_DIR = PROCESSED_DATA_DIR / "page_texts"
INDICES_DIR = PROCESSED_DATA_DIR / "indices"

# Modèles et Paramètres IA
TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
VISUAL_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32"
# Cross-encoder multilingue (même famille que BGE-M3, supporte le français)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Score sigmoïde minimal [0, 1] pour qu'un passage soit considéré comme preuve.
# Valeur prudente provisoire : à calibrer sur le jeu d'évaluation au Sprint 6.
RERANKER_MIN_SCORE = 0.05
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Modèle vision-langage (VLM) : il doit lire les schémas joints, donc pas de modèle texte seul.
# qwen2.5vl:3b (~3 Go) tient sur un PC 16 Go sans GPU dédié ; qwen2.5vl:7b pour la démo si la RAM le permet.
OLLAMA_MODEL = os.getenv("MAINTIDOC_OLLAMA_MODEL", "qwen2.5vl:3b")

# OCR (schémas techniques et pages scannées) — moteur RapidOCR/ONNX
OCR_ENABLED = os.getenv("MAINTIDOC_OCR", "1") != "0"
# Agrandissement avant reconnaissance : x2 restaure les espaces perdus sur le petit texte
OCR_UPSCALE = 2
# Lignes OCR sous ce seuil de confiance ignorées (bruit, flèches lues comme du texte)
OCR_MIN_CONFIDENCE = 0.5
# Texte OCR retenu seulement s'il contient au moins N caractères alphanumériques
OCR_MIN_ALNUM_CHARS = 4
# Page « scannée » : couche texte quasi vide ET une image couvrant l'essentiel de la page.
# Les deux conditions évitent de lancer l'OCR sur une page simplement blanche (ex. p.8).
OCR_PAGE_MAX_TEXT_CHARS = 100
OCR_PAGE_MIN_IMAGE_COVERAGE = 0.5
OCR_PAGE_DPI = 200

# Orchestration agentique (LangGraph) — bornes des boucles de contrôle.
# Toute boucle est bornée : un agent ne doit jamais pouvoir tourner indéfiniment.
AGENT_MAX_CLARIFICATIONS = 2      # questions de clarification posées au plus par conversation
AGENT_MAX_RETRIEVAL_ATTEMPTS = 2  # 1 recherche + 1 reformulation si aucune preuve
AGENT_MAX_GENERATION_ATTEMPTS = 2  # 1 génération + 1 correction si citations invalides
AGENT_RERANK_TOP_K = 4
# Checkpoints SQLite : une conversation en attente de clarification (et, au Sprint 4,
# d'approbation humaine) survit au redémarrage du notebook ou du serveur.
AGENT_CHECKPOINT_DB = PROCESSED_DATA_DIR / "checkpoints.sqlite"

# Métadonnées SAN
DOC_ID = "SAN-MAN-001"
TOTAL_EXPECTED_PAGES = 9
TOTAL_EXPECTED_SCHEMAS = 5

