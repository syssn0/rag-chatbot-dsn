"""
Ce script se lance UNE SEULE FOIS (ou a chaque ajout/modification de PDF).
Il transforme vos documents PDF bruts en une base de donnees vectorielle
(FAISS) que rag_chain.py pourra interroger rapidement.
Etape 1 du pipeline RAG : construction de l'index vectoriel.

Flux : PDF -> extraction du texte -> decoupage en chunks
       -> embeddings -> stockage dans un index FAISS local

A executer une seule fois (ou a chaque ajout/modification de documents) :
    python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config


def load_pdfs(docs_dir: Path) -> list:
    """Charge tous les PDF du dossier et garde la metadonnee de page."""
    all_documents = []
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouve dans {docs_dir}. "
            "Placez-y vos fichiers (reglement, CV, rapport de stage, etc.)."
        )

    for pdf_path in pdf_files:
        print(f"  Chargement : {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source_filename"] = pdf_path.name

        all_documents.extend(pages)

    return all_documents


def split_documents(documents: list) -> list:
    """Decoupe les documents en chunks avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(chunks: list) -> FAISS:
    """Calcule les embeddings et construit l'index FAISS."""
    print(f"  Modele d'embeddings : {config.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def main():
    print("1/3 - Chargement des PDF...")
    documents = load_pdfs(config.DOCS_DIR)
    print(f"   -> {len(documents)} page(s) chargee(s)")

    print("2/3 - Decoupage en chunks...")
    chunks = split_documents(documents)
    print(f"   -> {len(chunks)} chunk(s) cree(s)")

    print("3/3 - Calcul des embeddings et indexation FAISS...")
    vectorstore = build_index(chunks)

    config.INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(config.INDEX_DIR))
    print(f"\nIndex sauvegarde dans : {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
