"""Encodage vectoriel multimodal (BGE-M3 pour le texte, CLIP pour les schémas)."""
from typing import List, Union
from pathlib import Path
import numpy as np
from PIL import Image

class EmbeddingsManager:
    """Gestionnaire unifié des modèles d'embeddings."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._text_model = None
        self._clip_model = None

    @property
    def text_model(self):
        """Chargement paresseux (lazy loading) de BGE-M3 via HuggingFaceEmbedding."""
        if self._text_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._text_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=self.device)
        return self._text_model

    @property
    def clip_model(self):
        """Chargement paresseux de CLIP ViT-B/32."""
        if self._clip_model is None:
            from sentence_transformers import SentenceTransformer
            self._clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device=self.device)
        return self._clip_model

    def encode_text(self, text: str) -> List[float]:
        """Génère un embedding dense de 1024 dimensions pour une chaîne."""
        return self.text_model.get_text_embedding(text)

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes."""
        return [self.encode_text(t) for t in texts]

    def encode_image(self, image_input: Union[str, Path, Image.Image]) -> List[float]:
        """Génère un vecteur normalisé de 512 dimensions pour une image."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as im:
                rgb_im = im.convert("RGB")
                vec = self.clip_model.encode([rgb_im], normalize_embeddings=True)[0]
        else:
            rgb_im = image_input.convert("RGB")
            vec = self.clip_model.encode([rgb_im], normalize_embeddings=True)[0]
        return vec.tolist()

    def encode_visual_query(self, query: str) -> List[float]:
        """Encode une requête textuelle projetée dans l'espace latent CLIP (512-dim)."""
        vec = self.clip_model.encode([query], normalize_embeddings=True)[0]
        return vec.tolist()

