"""PDF-Chat – Vergleichs-App für den Stresstest.

Läuft, sieht gut aus. Teste sie mit dem Stresstest-Blatt.
Start (aus diesem Ordner): uv run streamlit run naive_app.py
Deployment: nur mit dem Secret RAG_MOCK = "1" (siehe README.md).
"""
import streamlit as st

import rag_core

EXAMPLE_QUESTIONS = ["Was ist RAG?", "Wie funktioniert Retrieval?", "Wann lohnt sich RAG?"]

st.set_page_config(page_title="Chat mit dem RAG-Leitfaden", page_icon="📄")


@st.cache_resource(show_spinner="Lade Index …")
def get_assistant():
    """Assistent nur einmal erzeugen – das Laden des Index dauert."""
    return rag_core.Assistant()


assistant = get_assistant()
st.session_state.setdefault("sources", [])  # Quellen je Antwort, parallel zu assistant.history


def chat_as_markdown() -> str:
    return "\n\n".join(f"**{m['role']}:** {m['content']}" for m in assistant.history)


def show_sources(sources):
    with st.expander(f"Quellen ({len(sources)})"):
        for s in sources:
            st.caption(f"S. {s['page']}: {s['text'][:200]}…")


with st.sidebar:
    st.header("Einstellungen")
    k = st.slider("Anzahl Quellen", 1, 5, 3)
    if st.button("Neuer Chat", width="stretch"):
        assistant.reset()
        st.session_state.sources = []
    st.download_button(
        "Chat herunterladen", chat_as_markdown(), "chat.md",
        icon=":material/download:", icon_position="right", width="stretch",
    )
    st.divider()
    stats = rag_core.stats()
    st.metric("Meine LLM-Aufrufe", stats["llm_calls"])
    st.metric("Index geladen (Server)", stats["index_loads"])

st.title("Chat mit dem RAG-Leitfaden")
st.caption("ⓘ Antworten stammen ausschließlich aus dem RAG-Leitfaden.")

if st.button("Dokument zusammenfassen"):
    st.info(assistant.summarize(), title="Zusammenfassung")

st.caption("Beispielfragen:")
with st.container(horizontal=True):
    for example in EXAMPLE_QUESTIONS:
        if st.button(example):
            st.session_state.example = example

answers = 0
for msg in assistant.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if answers < len(st.session_state.sources):
                show_sources(st.session_state.sources[answers])
            answers += 1

question = st.chat_input("Deine Frage zum Leitfaden") or st.session_state.get("example")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer = assistant.ask(question, k=k)
        st.session_state.sources.append(assistant.last_sources)
        st.markdown(answer)
        show_sources(assistant.last_sources)
