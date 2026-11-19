"""PDF-Chat – Referenzlösung für den Workshop (besteht den Stresstest).

Start (aus diesem Ordner): uv run streamlit run app.py
Die Kommentare [K1]–[K4] (Kernchecks) und [B1]–[B4] (Bonus) verweisen auf das Stresstest-Blatt.
Gleicher Funktionsumfang wie naive_app.py (und wie wireframe.png) – nur framework-korrekt.
"""
import streamlit as st

import rag_core

EXAMPLE_QUESTIONS = ["Was ist RAG?", "Wie funktioniert Retrieval?", "Wann lohnt sich RAG?"]

st.set_page_config(page_title="Chat mit dem RAG-Leitfaden", page_icon="📄")


# [K1, K3] Den teuren, nicht-persönlichen Teil (Index) einmal pro Server laden und teilen.
@st.cache_resource(show_spinner="Lade Index …")
def get_index():
    return rag_core.load_index()


# [K3] Der Assistent hat ein Gedächtnis = Nutzerdaten -> pro Browser-Tab in session_state,
# aber mit dem geteilten Index. Den ganzen Assistenten zu cachen hieße: ein Gedächtnis für alle.
if "assistant" not in st.session_state:
    st.session_state.assistant = rag_core.Assistant(index=get_index())
assistant = st.session_state.assistant
st.session_state.setdefault("sources", [])  # Quellen je Antwort, parallel zu assistant.history
st.session_state.setdefault("summary", None)
st.session_state.setdefault("my_calls", 0)  # [K4] eigener Zähler statt rag_core.stats() (= alle Nutzer)


def new_chat():
    assistant.reset()
    st.session_state.sources = []
    st.session_state.summary = None


def pick_example(question):
    # [K1] Callback merkt die Frage für genau einen Durchlauf; sie wird unten mit pop() verbraucht.
    st.session_state.pending = question


def chat_as_markdown() -> str:
    return "\n\n".join(f"**{m['role']}:** {m['content']}" for m in assistant.history)


def show_sources(sources):
    with st.expander(f"Quellen ({len(sources)})"):
        for s in sources:
            st.caption(f"S. {s['page']}: {s['text'][:200]}…")


with st.sidebar:
    st.header("Einstellungen")
    k = st.slider("Anzahl Quellen", 1, 5, 3)
    st.button("Neuer Chat", on_click=new_chat, width="stretch")
    st.download_button(
        "Chat herunterladen", chat_as_markdown(), "chat.md",
        icon=":material/download:", icon_position="right", width="stretch",
    )
    st.divider()
    counter = st.empty()  # [B3] Platz reservieren, Werte erst am Ende des Skripts eintragen

st.title("Chat mit dem RAG-Leitfaden")
st.caption("ⓘ Antworten stammen ausschließlich aus dem RAG-Leitfaden.")

# [B1] Ein Button ist nur für einen Rerun True – das Ergebnis deshalb merken.
if st.button("Dokument zusammenfassen"):
    with st.spinner("Fasse zusammen …"):
        st.session_state.summary = assistant.summarize()
        st.session_state.my_calls += 1
if st.session_state.summary:
    st.info(st.session_state.summary, title="Zusammenfassung")

st.caption("Beispielfragen:")
with st.container(horizontal=True):
    for example in EXAMPLE_QUESTIONS:
        st.button(example, on_click=pick_example, args=(example,))

answers = 0
for msg in assistant.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if answers < len(st.session_state.sources):
                show_sources(st.session_state.sources[answers])
            answers += 1

# [K1] st.chat_input hat nur im Rerun direkt nach dem Absenden einen Wert; eine Beispielfrage
# wird mit pop() genau einmal verbraucht. Der Slider löst deshalb keinen LLM-Aufruf aus.
question = st.chat_input("Deine Frage zum Leitfaden") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        # [K2] Eine Interaktion während der Antwort bricht das Skript sofort ab – auch mitten
        # in ask(). Der Verlauf wird nur in rag_core geführt und erst nach der Antwort ergänzt:
        # Frage und Antwort fehlen dann beide (konsistent), statt einer Frage ohne Antwort.
        # Der LLM-Aufruf ist trotzdem bezahlt. Ganz verhindern: Widgets währenddessen deaktivieren.
        # [B4] Spinner zeigt, dass gearbeitet wird.
        with st.spinner("Denke nach …"):
            answer = assistant.ask(question, k=k)
            st.session_state.sources.append(assistant.last_sources)
            st.session_state.my_calls += 1
        st.markdown(answer)
        show_sources(assistant.last_sources)

with counter.container():
    st.metric("Meine LLM-Aufrufe", st.session_state.my_calls)
    # Index-Ladevorgänge sind eine Server-Kennzahl – dafür ist rag_core.stats() richtig.
    st.metric("Index geladen (Server)", rag_core.stats()["index_loads"])
# [Deployment] Keys kommen aus der .env (rag_core lädt sie) – nie aus dem Code.
