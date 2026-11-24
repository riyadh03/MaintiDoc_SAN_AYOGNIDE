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
OLLAMA_MODEL = "qwen3.8:27b"
OLLAMA_FALLBACK_MODEL = "qwen2.5:14b"

# Métadonnées SAN
DOC_ID = "SAN-MAN-001"
TOTAL_EXPECTED_PAGES = 9
TOTAL_EXPECTED_SCHEMAS = 5

