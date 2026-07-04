"""
Script de diagnostic : affiche le texte brut extrait de chaque PDF,
pour verifier que l'extraction et le decoupage se sont bien passes.
"""

from langchain_community.document_loaders import PyPDFLoader
import config

pdf_files = sorted(config.DOCS_DIR.glob("*.pdf"))

for pdf_path in pdf_files:
    print(f"\n{'='*60}")
    print(f"FICHIER : {pdf_path.name}")
    print(f"{'='*60}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"Nombre de pages : {len(pages)}")
    for i, page in enumerate(pages):
        print(f"\n--- Page {i+1} (extrait, 300 premiers caracteres) ---")
        print(page.page_content[:300])
