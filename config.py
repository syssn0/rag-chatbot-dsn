"""
Configuration centrale du chatbot RAG.
Ce fichier centralise TOUS les parametres ajustables du projet.
Objectif : pouvoir changer le comportement du chatbot (modele, taille des
chunks, nombre de resultats...) sans toucher au code des autres fichiers.
"""

from pathlib import Path

# Dossiers
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "faiss_index"

# Decoupage du texte (chunking)
CHUNK_SIZE = 1000          # nombre de caracteres par chunk
CHUNK_OVERLAP = 150        # chevauchement entre deux chunks consecutifs

# Modele d'embeddings (HuggingFace, tourne en local, leger et rapide)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modele LLM local via Ollama
# "phi3" : modele Microsoft, 3.8 milliards de parametre 
LLM_MODEL = "phi3"

# Nombre de passages pertinents a recuperer pour chaque question
TOP_K = 4
