"""PDF-Chat – Referenzlösung für Tag 5 (besteht den Stresstest).

Start (aus dem Repo-Root): streamlit run unsere_skripte/tag5_app/fixed_app.py
Die Kommentare [K1]–[K4] (Kernchecks) und [B1]–[B4] (Bonus) verweisen auf das Stresstest-Blatt.
Gleicher Funktionsumfang wie naive_app.py – nur framework-korrekt.
"""
import streamlit as st

import rag_core

st.title("📄 Chat mit dem RAG-Leitfaden - Let's go")


# [K1, K3] Den teuren, nicht-persönlichen Teil (Index) einmal pro Server laden und teilen.
@st.cache_resource(show_spinner="Lade Index …")
def get_index():
    return rag_core.load_index()


# [K3] Der Assistent hat ein Gedächtnis = Nutzerdaten -> pro Browser-Tab in session_state,
# aber mit dem geteilten Index. Den ganzen Assistenten zu cachen hieße: ein Gedächtnis für alle.
if "assistant" not in st.session_state:
    st.session_state.assistant = rag_core.Assistant(index=get_index())
assistant = st.session_state.assistant
st.session_state.setdefault("summary", None)
st.session_state.setdefault("my_calls", 0)  # [K4] eigener Zähler statt rag_core.stats() (= alle Nutzer)

with st.sidebar:
    k = st.slider("Anzahl Quellen", 1, 5, 3)
    counter = st.empty()  # [B3] Platz reservieren, Wert erst am Ende des Skripts eintragen

# [B1] Ein Button ist nur für einen Rerun True – das Ergebnis deshalb merken.
if st.button("Dokument zusammenfassen", key="summary_btn"):
    with st.spinner("Fasse zusammen …"):
        st.session_state.summary = assistant.summarize()
        st.session_state.my_calls += 1
if st.session_state.summary:
    st.info(st.session_state.summary)

for msg in assistant.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# [K1] st.chat_input hat nur im Rerun direkt nach dem Absenden einen Wert.
if question := st.chat_input("Deine Frage zum Dokument"):
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
            st.session_state.my_calls += 1
        st.markdown(answer)

counter.metric("Meine LLM-Aufrufe", st.session_state.my_calls)
# [Deployment] Keys kommen aus der .env (rag_core lädt sie) – nie aus dem Code.
