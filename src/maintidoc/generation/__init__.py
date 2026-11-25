"""Module de génération LLM/VLM."""
from .ollama_client import ask_maintidoc_vlm, validate_citations

__all__ = ["ask_maintidoc_vlm", "validate_citations"]

