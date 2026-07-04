"""
Etape 3 du pipeline RAG : interface utilisateur avec historique de chat.

Lancer avec :
    streamlit run app_streamlit.py
"""

import streamlit as st

import config
from access_control import ROLES
from rag_chain import load_vectorstore, answer_question

st.set_page_config(page_title="Chatbot documentaire RAG", page_icon="💬")

st.title("Chatbot documentaire (RAG local)")
st.caption("Toutes les reponses sont generees a partir de vos PDF, en local, sans appel externe.")

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

try:
    vectorstore = get_vectorstore()
except Exception as e:
    st.error(
        "Index introuvable ou illisible. "
        "Avez-vous bien lance `python ingest.py` au prealable ?"
    )
    st.exception(e)
    st.stop()

# Role choisi dans la barre laterale, pour ne pas encombrer le fil de chat
with st.sidebar:
    st.header("Parametres")
    role = st.selectbox("Votre role", ROLES, index=ROLES.index("employe"))
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialisation de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique existant
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources utilisees"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Zone de saisie en bas de page, comme un vrai chat
question = st.chat_input("Posez votre question...")

if question:
    # Afficher et enregistrer la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generer et afficher la reponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche et generation de la reponse..."):
            result = answer_question(vectorstore, question, role)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources utilisees"):
                for source in result["sources"]:
                    st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

