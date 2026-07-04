"""
RAG_CHAIN.PY - Etape 2 du pipeline : repondre a une question

C'est le coeur du systeme RAG. Ce fichier orchestre les 4 etapes cles :

    1. Recherche vectorielle : trouver les chunks les plus proches
       semantiquement de la question posee (via FAISS)
    2. Filtrage par role : ne garder que les chunks autorises
       pour la personne qui pose la question (via access_control.py)
    3. Construction du prompt : assembler le contexte + la question
       dans un format que le LLM comprend
    4. Generation : appeler le LLM local (Ollama/phi3) pour produire
       la reponse finale, avec les sources utilisees

Peut aussi etre lance seul en ligne de commande : python rag_chain.py

"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# OllamaLLM : classe LangChain qui envoie des requetes au serveur
# Ollama local (qui tourne sur http://127.0.0.1:11434 par defaut)
from langchain_ollama import OllamaLLM

# PromptTemplate : permet de construire un prompt avec des "variables"
# ({context} et {question}) qui seront remplacees automatiquement
from langchain_core.prompts import PromptTemplate

import config
from access_control import filter_documents_by_role

# C'est l'instruction envoyee au LLM a CHAQUE question. Le point cle :
# "repond UNIQUEMENT a partir du contexte" -> c'est cette phrase qui
# empeche le LLM d'utiliser ses connaissances generales (entrainement)
# et le force a ne se baser que sur les documents fournis. C'est ce qui
# garantit que le systeme ne repond pas, par exemple, a "qui est Elon Musk"
# alors que ce n'est dans aucun document.
PROMPT_TEMPLATE = """Tu es un assistant qui repond aux questions UNIQUEMENT a partir du contexte fourni ci-dessous.
Si l'information ne se trouve pas dans le contexte, dis clairement que tu ne sais pas.
Reponds en francais, de facon claire et concise.

Contexte :
{context}

Question : {question}

Reponse :"""


def load_vectorstore() -> FAISS:
    """
    Recharge l'index FAISS depuis le disque (cree par ingest.py).

    Important : on doit utiliser EXACTEMENT le meme modele d'embeddings
    que celui utilise pour construire l'index (dans ingest.py), sinon
    les vecteurs ne seraient pas comparables entre eux.

    allow_dangerous_deserialization=True : necessaire car FAISS utilise
    pickle pour charger les metadonnees. Sans danger ici car c'est
    NOUS qui avons cree ce fichier (pas une source externe non fiable).
    """
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_relevant_chunks(vectorstore: FAISS, question: str, role: str, k: int = None):
    """
    Etape RECHERCHE + FILTRAGE.

    Astuce importante : on recupere k*3 chunks au depart (au lieu de k),
    PUIS on filtre par role, PUIS on ne garde que les k premiers.
    Pourquoi ? Si on demandait directement k chunks et qu'on filtrait
    apres, on pourrait se retrouver avec moins de k chunks au final
    (si certains des k etaient interdits). En sur-echantillonnant
    d'abord, on garde de la marge pour que le filtrage par role
    ne nous prive pas de contenu pertinent autorise.
    """
    k = k or config.TOP_K

    # similarity_search : calcule la distance entre l'embedding de la
    # question et tous les embeddings de l'index, renvoie les k*3 plus proches
    raw_results = vectorstore.similarity_search(question, k=k * 3)

    # Filtrage par role : c'est ICI que le controle d'acces s'applique
    # reellement, AVANT que quoi que ce soit soit envoye au LLM
    filtered_results = filter_documents_by_role(raw_results, role)

    return filtered_results[:k]


def format_sources(chunks: list) -> list:
    """
    Construit la liste lisible des sources utilisees, avec numero
    de page, pour l'afficher a l'utilisateur (transparence/tracabilite,
    exigence explicite du contexte hospitalier).
    """
    sources = []
    for chunk in chunks:
        filename = chunk.metadata.get("source_filename", "inconnu")
        page = chunk.metadata.get("page", None)
        if page is not None:
            # +1 car PyPDFLoader numerote les pages a partir de 0,
            # alors qu'un humain compte les pages a partir de 1
            sources.append(f"{filename} (page {page + 1})")
        else:
            sources.append(filename)

    # Dedoublonnage tout en gardant l'ordre d'apparition
    # (plusieurs chunks peuvent venir de la meme page)
    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)
    return unique_sources


def answer_question(vectorstore: FAISS, question: str, role: str = "employe") -> dict:
    """
    Fonction principale appelee depuis l'interface (Streamlit ou CLI).

    Retourne un dictionnaire avec :
        - "answer"  : la reponse textuelle generee par le LLM
        - "sources" : la liste des documents/pages utilises
    """
    relevant_chunks = retrieve_relevant_chunks(vectorstore, question, role)

    # Cas ou AUCUN chunk autorise n'a ete trouve : soit la question
    # n'a pas de reponse dans les documents, soit le role n'a acces
    # a aucun document pertinent. On ne contacte meme pas le LLM,
    # on renvoie directement un message clair.
    if not relevant_chunks:
        return {
            "answer": "Je n'ai trouve aucune information autorisee pour repondre a cette question.",
            "sources": [],
        }

    # On concatene le texte de tous les chunks retenus pour former
    # le "contexte" qui sera insere dans le prompt
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    # .format() remplace {context} et {question} par les vraies valeurs
    formatted_prompt = prompt.format(context=context, question=question)

    # Connexion au serveur Ollama local et envoi du prompt complet
    llm = OllamaLLM(model=config.LLM_MODEL)
    answer = llm.invoke(formatted_prompt)

    return {
        "answer": answer.strip(),
        "sources": format_sources(relevant_chunks),
    }


# Permet de tester ce fichier seul, en ligne de commande,
# sans passer par l'interface Streamlit
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    question = input("Votre question : ")
    role = input("Votre role (admin/employe/etudiant) : ").strip() or "employe"

    result = answer_question(vectorstore, question, role)
    print("\nReponse :", result["answer"])
    print("\nSources utilisees :")
    for source in result["sources"]:
        print(" -", source)
