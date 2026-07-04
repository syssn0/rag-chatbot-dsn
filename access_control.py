"""
Controle d'acces simplifie par role.

Chaque document est associe a une liste de roles autorises a le consulter.
Le role "admin" voit tout par defaut.
"""

# A adapter selon vos propres fichiers PDF.
# Cle = nom exact du fichier dans le dossier documents/
# Valeur = liste des roles autorises
DOCUMENT_PERMISSIONS = {
    "reglement_interieur.pdf": ["admin", "employe"],
    "guide_etudiant.pdf": ["admin", "employe", "etudiant"],
    "documentation_python.pdf": ["admin", "employe", "etudiant"],
    "documentation_docker.pdf": ["admin", "employe", "etudiant"],
    "rapport_de_stage.pdf": ["admin", "employe"],
    "cv.pdf": ["admin", "employe", "etudiant"],
}

ROLES = ["admin", "employe", "etudiant"]


def is_authorized(source_filename: str, role: str) -> bool:
    """
    Verifie si un role donne a le droit de consulter un document.
    Si le document n'est pas repertorie, on le refuse par defaut.
    """
    if role == "admin":
        return True
    allowed_roles = DOCUMENT_PERMISSIONS.get(source_filename, [])
    return role in allowed_roles


def filter_documents_by_role(documents: list, role: str) -> list:
    """
    Filtre une liste de documents LangChain (avec metadata['source_filename'])
    pour ne garder que ceux autorises pour le role donne.
    """
    filtered = []
    for doc in documents:
        source_filename = doc.metadata.get("source_filename", "")
        if is_authorized(source_filename, role):
            filtered.append(doc)
    return filtered
