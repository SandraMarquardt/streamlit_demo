"""PDF-Chat – Fallback-App für Tag 5.

Läuft, sieht gut aus. Teste sie mit dem Stresstest-Blatt.
Start (aus dem Repo-Root): streamlit run unsere_skripte/tag5_app/naive_app.py
"""
import streamlit as st

import rag_core

# OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # zum schnellen Testen

st.title("📄 Chat mit dem RAG-Leitfaden")


@st.cache_resource
def get_assistant():
    """Assistent nur einmal erzeugen – das Laden des Index dauert."""
    return rag_core.Assistant()


assistant = get_assistant()

with st.sidebar:
    k = st.slider("Anzahl Quellen", 1, 5, 3)
    st.metric("Meine LLM-Aufrufe", rag_core.stats()["llm_calls"])

if st.button("Dokument zusammenfassen", key="summary_btn"):
    st.info(assistant.summarize())

for msg in assistant.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.text_input("Deine Frage", key="question")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(assistant.ask(question, k=k))
