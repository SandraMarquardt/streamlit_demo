import logging
import os

import streamlit as st

import rag_core

log = logging.getLogger(__name__)
API_KEYS = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")
SETTINGS = ("RAG_PROVIDER", "RAG_OPENROUTER_MODEL", "RAG_ANTHROPIC_MODEL")
EXAMPLE_QUESTIONS = ["Was ist RAG?", "Wie funktioniert Retrieval?", "Wann lohnt sich RAG?"]
SOURCE_CHARS = 300


def secrets_to_env():
    # rag_core liest nur Umgebungsvariablen. Auf dem Server kommen die Keys aus den
    # Streamlit-Secrets, lokal optional aus .streamlit/secrets.toml.
    try:
        secrets = st.secrets
        for key in API_KEYS + SETTINGS:
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
    except Exception:
        pass


secrets_to_env()


@st.cache_resource(show_spinner="Leitfaden wird eingelesen …")
def get_index():
    return rag_core.load_index()


def new_chat():
    st.session_state.messages = []
    st.session_state.summary = None
    st.session_state.assistant.reset()


def render_sources(sources):
    with st.expander(f"Quellen ({len(sources)})"):
        for s in sources:
            st.markdown(f"**Seite {s['page']}**")
            st.caption(s["text"][:SOURCE_CHARS] + ("…" if len(s["text"]) > SOURCE_CHARS else ""))


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption("Quelle: RAG-Leitfaden")
            render_sources(msg["sources"])


def build_markdown():
    lines = ["# Chat mit dem RAG-Leitfaden", ""]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lines += [f"**Du:** {msg['content']}", ""]
        else:
            lines += [f"**KI:** {msg['content']}", "", "*Quelle: RAG-Leitfaden*", "", "Quellen:"]
            for s in msg["sources"]:
                text = s["text"][:SOURCE_CHARS] + ("…" if len(s["text"]) > SOURCE_CHARS else "")
                lines.append(f"- Seite {s['page']}: {text}")
            lines.append("")
    return "\n".join(lines)


def answer(question, k):
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            text = st.write_stream(st.session_state.assistant.ask_stream(question, k=k))
            sources = st.session_state.assistant.last_sources
            st.caption("Quelle: RAG-Leitfaden")
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
        except Exception:
            log.exception("Antwort fehlgeschlagen")
            st.error("Die Antwort konnte gerade nicht erstellt werden. Bitte versuche es erneut.")
            st.session_state.messages.pop()


st.set_page_config(page_title="Chat mit dem RAG-Leitfaden", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.summary = None

try:
    if "assistant" not in st.session_state:
        st.session_state.assistant = rag_core.Assistant(get_index())
except Exception:
    log.exception("Leitfaden konnte nicht geladen werden")
    st.error("Der Leitfaden konnte nicht geladen werden. Bitte die Serverlogs prüfen.")
    st.stop()

with st.sidebar:
    st.header("Einstellungen")
    k = st.slider("Anzahl Quellen", min_value=1, max_value=5, value=3)
    if st.button("Neuer Chat", width="stretch"):
        new_chat()
        st.rerun()
    st.download_button(
        "Chat herunterladen ↓",
        data=build_markdown(),
        file_name="chat_rag_leitfaden.md",
        mime="text/markdown",
        disabled=not st.session_state.messages,
        width="stretch",
    )

st.title("Chat mit dem RAG-Leitfaden")
st.caption("ⓘ Antworten stammen ausschließlich aus dem RAG-Leitfaden.")

if rag_core.provider() == "mock":
    st.warning(
        "Kein API-Key konfiguriert. Die Antworten sind nur Platzhalter. "
        "Bitte OPENROUTER_API_KEY oder ANTHROPIC_API_KEY in den Secrets hinterlegen."
    )

if st.button("Dokument zusammenfassen"):
    with st.spinner("Zusammenfassung wird erstellt …"):
        try:
            st.session_state.summary = st.session_state.assistant.summarize()
        except Exception:
            log.exception("Zusammenfassung fehlgeschlagen")
            st.error("Die Zusammenfassung konnte gerade nicht erstellt werden. Bitte versuche es erneut.")

if st.session_state.summary:
    with st.container(border=True):
        st.markdown("**Zusammenfassung**")
        st.markdown(st.session_state.summary)

st.caption("Beispielfragen:")
clicked = None
for col, q in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
    with col:
        if st.button(q, key=f"ex_{q}", width="stretch"):
            clicked = q

for msg in st.session_state.messages:
    render_message(msg)

prompt = st.chat_input("Deine Frage zum Leitfaden")
question = clicked or prompt
if question:
    answer(question, k)
