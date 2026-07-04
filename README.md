# Chatbot Documentaire RAG Local

Chatbot de type RAG (Retrieval-Augmented Generation) fonctionnant **entièrement en local**,
sans aucun appel à une API externe, avec **contrôle d'accès aux documents par rôle utilisateur**.

Projet personnel explorant les briques techniques d'un système RAG documentaire
avec gestion des droits d'accès.

## Fonctionnalités

- Extraction et indexation de documents PDF (LangChain + FAISS)
- Génération de réponses via un LLM local (Ollama / Mistral)
- Citation des sources utilisées avec numéro de page
- Contrôle d'accès par rôle (admin / employe / etudiant) : chaque document
  n'est visible que par les rôles autorisés, appliqué au niveau de la
  recherche elle-même (pas seulement de l'affichage)
- Interface de chat avec historique (Streamlit)

## Architecture
PDF → extraction (PyPDFLoader) → chunking → embeddings (HuggingFace)
→ indexation (FAISS) → recherche → filtrage par rôle
→ génération (LLM local) → réponse + sources

## Stack technique

- Python
- LangChain
- FAISS (base vectorielle)
- Ollama (exécution du LLM en local)
- Streamlit (interface)
- sentence-transformers (embeddings)

## Installation

```bash
pip install -r requirements.txt
```

Installer Ollama ([ollama.com](https://ollama.com)) puis télécharger un modèle :

```bash
ollama pull phi3
```

## Utilisation

1. Placer ses PDF dans le dossier `documents/`
2. Adapter les permissions dans `access_control.py`
3. Construire l'index :
```bash
   python ingest.py
```
4. Lancer l'interface :
```bash
   streamlit run app_streamlit.py
```

## Structure du projet

| Fichier | Rôle |
|---|---|
| `config.py` | Paramètres centralisés (modèles, taille des chunks) |
| `access_control.py` | Gestion des rôles et permissions par document |
| `ingest.py` | Extraction PDF, chunking, indexation FAISS |
| `rag_chain.py` | Recherche, filtrage par rôle, génération de la réponse |
| `app_streamlit.py` | Interface web (chat) |

## Auteur
Sokhna Yaye Safietou Sy Niang — Élève ingénieure en informatique, ISEN Méditerranée
