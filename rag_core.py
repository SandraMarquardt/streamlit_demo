"""rag_core – RAG-Backend für den Workshop: Fragen zu einem PDF beantworten.

Nicht ändern – nur importieren.

    import rag_core
    assistant = rag_core.Assistant()
    print(assistant.ask("Was ist RAG?"))

Retrieval per TF-IDF (numpy), LLM über OpenRouter oder Anthropic (Key aus der .env im Projektordner).

Umgebungsvariablen (optional, z. B. in der .env):
    RAG_MOCK=1              Antworten simulieren statt LLM aufzurufen
    RAG_PROVIDER            openrouter | anthropic | mock   (Standard: automatisch)
    RAG_OPENROUTER_MODEL    Standard: z-ai/glm-5.3-flash
    RAG_ANTHROPIC_MODEL     Standard: claude-haiku-5-5
    RAG_LOAD_DELAY          Ladezeit eines Embedding-Modells simulieren, in Sekunden (Standard 2)
    RAG_MOCK_DELAY          Antwortzeit im Mock-Modus, in Sekunden (Standard 3)

Kursbezug – die RAG-Schritte aus Modul M07 und wo sie hier stecken:
    1. Daten laden    _build()      ~ PyPDFLoader in scripts/M07_VectorDB_RAG/V01_DataLoading.py
    2. Chunking       _chunk()      ~ CharacterTextSplitter in M07_VectorDB_RAG/20_chunking/10_fixed_size_chunking.py
    3. Vektorisieren  _build()      ~ Embeddings in M07_VectorDB_RAG/30_embedding/ (hier: TF-IDF statt Embeddings)
    4. Speichern      Index         ~ Chroma in M07_VectorDB_RAG/40_vector_store/10_chromadb_store.py
    5. Retrieval      retrieve()    ~ retriever.invoke() in M07_VectorDB_RAG/50_retrieve_data/10_chromadb_retrieval.py
    6. Augmentation   _messages()   ~ Prompt mit Kontext in M07_VectorDB_RAG/RAG02_app.py
    7. Generation     _complete()   ~ model.invoke() aus M01_Einführung
Die Stellen sind unten mit "Kursbezug:" markiert.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = HERE / "data" / "rag_leitfaden.pdf"
# Kursbezug: dieselben Stellschrauben wie beim Chunking in M07, z. B.
# CHUNK_SIZE = 1000 / CHUNK_OVERLAP = 200 in M07_VectorDB_RAG/90_capstone_project/10_data_prep.py.
# Die Überlappung sorgt dafür, dass ein Satz an der Chunk-Grenze nicht verloren geht.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUMMARY_PROMPT = "Fasse das Dokument in 5 Stichpunkten zusammen."


def _load_env() -> Path | None:
    """Sucht die .env von diesem Ordner aufwärts (also auch die im Projektordner)."""
    # Kursbezug: entspricht load_dotenv(find_dotenv(usecwd=True)) aus den Skripten in
    # M01_Einführung (z. B. 10_model_chat_openrouter.py). Hier wird ab dem Ordner dieser
    # Datei gesucht statt ab dem aktuellen Arbeitsverzeichnis – so findet auch eine
    # Streamlit-App, die aus einem anderen Ordner gestartet wird, die .env.
    for folder in [HERE, *HERE.parents]:
        env_file = folder / ".env"
        if env_file.is_file():
            from dotenv import load_dotenv

            load_dotenv(env_file)  # überschreibt keine bereits gesetzten Variablen
            return env_file
    return None


ENV_FILE = _load_env()

# Streamlit bedient mehrere Nutzer/Tabs in parallelen Threads. Der Lock stellt sicher,
# dass das LLM-Objekt nur einmal erzeugt wird (ähnlich wie @st.cache_resource).
_lock = threading.Lock()
_llm = None


# --------------------------------------------------------------------------- index

# TF-IDF – kurz erklärt
# ---------------------
# Was ist das?
#   TF-IDF = "Term Frequency × Inverse Document Frequency". Jeder Chunk wird zu einem
#   Vektor mit einer Spalte pro Wort im gesamten Dokument (Vokabular).
#     - TF:  Wie oft kommt das Wort in diesem Chunk vor?
#     - IDF: Wie selten ist das Wort über alle Chunks? Seltene Wörter ("Chunking",
#            "Retriever") bekommen viel Gewicht, Allerweltswörter ("der", "und") wenig.
#   Frage und Chunks werden gleich vektorisiert und per Kosinus-Ähnlichkeit verglichen –
#   die k ähnlichsten Chunks sind die Quellen.
#
# Warum für diese Demo?
#   - Kein API-Key, kein Modell-Download, kein PyTorch: nur numpy. Die App startet sofort
#     und lässt sich ohne zusätzliche Abhängigkeiten in die Cloud deployen.
#   - Schnell (Millisekunden) und deterministisch: gleiche Frage -> gleiche Quellen.
#   - Nachvollziehbar: Man sieht, *warum* ein Chunk passt – er enthält die Wörter der Frage.
#
# Vergleich mit den Verfahren aus scripts/
#   - Gleiches Prinzip wie bei Embeddings: Text -> Vektor -> Kosinus-Ähnlichkeit -> Top-k
#     (vgl. cosine_similarity in M01_Einführung/45_semantic_router.py).
#   - Embeddings (M07_VectorDB_RAG/30_embedding/: Word2Vec/GloVe, SentenceTransformer,
#     HuggingFaceEndpointEmbeddings) erfassen die *Bedeutung*: "Auto" und "Fahrzeug" liegen
#     nah beieinander, sogar über Sprachen hinweg (20_sentence_similarity.py). TF-IDF
#     vergleicht nur *exakte Wörter* – Synonyme oder eine deutsche Frage zu einem englischen
#     PDF finden deutlich schlechtere Treffer.
#   - Embedding-Vektoren sind dicht (z. B. 384 Zahlen bei all-MiniLM-L6-v2, alle ≠ 0).
#     TF-IDF-Vektoren haben so viele Spalten wie das Vokabular Wörter hat (beim Demo-PDF
#     ca. 380) und sind fast überall 0.
#   - Vector Stores (Chroma, Pinecone in M07_VectorDB_RAG/40_vector_store/) speichern die
#     Vektoren dauerhaft und skalieren auf Millionen Chunks. Hier liegt einfach eine
#     numpy-Matrix im Arbeitsspeicher – für ein einzelnes PDF völlig ausreichend.
#   - In der Praxis wird TF-IDF (bzw. der Nachfolger BM25) oft mit Embeddings kombiniert
#     ("Hybrid Search"): exakte Begriffe + Bedeutung.

@dataclass
class Index:
    # Kursbezug: übernimmt die Rolle des Vector Stores (z. B. Chroma in
    # M07_VectorDB_RAG/40_vector_store/10_chromadb_store.py) – Chunks + ihre Vektoren.
    name: str
    chunks: list[dict]  # {"text": str, "page": int}  ~ Document(page_content=..., metadata={"page": ...})
    vocab: dict[str, int]  # Wort -> Spaltennummer in der Matrix
    idf: np.ndarray  # Gewicht pro Wort (seltene Wörter = hohes Gewicht)
    matrix: np.ndarray  # L2-normalisierte TF-IDF-Zeilen (eine Zeile pro Chunk)


def _tokens(text: str) -> list[str]:
    # Zerlegt Text in kleingeschriebene Wörter. Kursbezug: ein sehr einfacher Tokenizer –
    # vgl. AutoTokenizer in M07_VectorDB_RAG/V02_DataChunking.py, der Text in Subwort-Tokens zerlegt.
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


def _chunk(pages: list[tuple[int, str]]) -> list[dict]:
    # Kursbezug: Fixed-Size-Chunking wie CharacterTextSplitter(chunk_size=..., chunk_overlap=...)
    # in M07_VectorDB_RAG/20_chunking/10_fixed_size_chunking.py. Unterschied: Hier wird ohne
    # Trennzeichen geschnitten, ein Chunk kann also mitten im Wort beginnen. Der
    # RecursiveCharacterTextSplitter (RAG01_data_prep.py) vermeidet das über "separators".
    # Die Seitennummer wird mitgespeichert – wie die Metadaten eines Document.
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page_no, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE]
            if piece.strip():
                chunks.append({"text": piece, "page": page_no})
            if start + CHUNK_SIZE >= len(text):
                break
    return chunks


def _build(pdf_source, name: str) -> Index:
    # Simuliert die Ladezeit, die ein echtes Embedding-Modell (z. B. SentenceTransformer in
    # M07_VectorDB_RAG/30_embedding/20_sentence_similarity.py) bräuchte. TF-IDF selbst ist
    # sofort fertig – die Pause macht Lade-Spinner und Caching in der App sichtbar.
    time.sleep(float(os.getenv("RAG_LOAD_DELAY", "2")))  # simuliert Embedding-Modell laden

    # 1. Daten laden. Kursbezug: PyPDFLoader (V01_DataLoading.py) nutzt intern ebenfalls
    #    pypdf und liefert ein Document pro Seite. Achtung: PyPDFLoader zählt Seiten ab 0, hier ab 1.
    reader = PdfReader(pdf_source)
    pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    # 2. Chunking
    chunks = _chunk(pages)
    if not chunks:
        raise ValueError(f"Kein Text im PDF gefunden: {name}")

    # 3. Vektorisieren mit TF-IDF (Erklärung siehe oben). Entspricht
    #    embeddings_model.embed_documents(texts) in M07_VectorDB_RAG/V03_Embeddings.py.
    docs = [_tokens(c["text"]) for c in chunks]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    tf = np.zeros((len(docs), len(vocab)))  # TF: Wortzählung, eine Zeile pro Chunk
    for row, doc in enumerate(docs):
        for t in doc:
            tf[row, vocab[t]] += 1
    df = (tf > 0).sum(axis=0)  # in wie vielen Chunks kommt jedes Wort vor?
    idf = np.log((1 + len(docs)) / (1 + df)) + 1  # IDF: selten -> groß (geglättet wie in scikit-learn)
    matrix = tf * idf
    # Jede Zeile auf Länge 1 bringen -> später ist Skalarprodukt = Kosinus-Ähnlichkeit
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    # 4. "Speichern": alles landet im Index-Objekt im Arbeitsspeicher (statt in Chroma)
    return Index(name=name, chunks=chunks, vocab=vocab, idf=idf, matrix=matrix)


def load_index(pdf_path: str | Path | None = None) -> Index:
    """Liest ein PDF ein und baut den Suchindex (Standard: data/rag_leitfaden.pdf)."""
    path = Path(pdf_path) if pdf_path else DEFAULT_PDF
    return _build(str(path), path.name)


def index_from_bytes(data: bytes, name: str = "upload.pdf") -> Index:
    """Wie load_index(), für ein hochgeladenes PDF (Bytes)."""
    # Für st.file_uploader: Die Datei kommt als Bytes, nicht als Pfad auf der Festplatte.
    return _build(BytesIO(data), name)


def retrieve(index: Index, question: str, k: int = 3) -> list[dict]:
    """Die k passendsten Abschnitte: [{"text", "page", "score"}, ...]."""
    # 5. Retrieval. Kursbezug: entspricht db.similarity_search_with_score(query, k=...) in
    #    M07_VectorDB_RAG/90_capstone_project/app.py bzw. retriever.invoke(query)[:3] in RAG02_app.py.
    #    Die Frage wird genauso vektorisiert wie die Chunks (vgl. embed_query vs. embed_documents
    #    in M01_Einführung/45_semantic_router.py).
    q = np.zeros(len(index.vocab))
    for t in _tokens(question):
        if t in index.vocab:  # Wörter, die nicht im PDF vorkommen, werden ignoriert
            q[index.vocab[t]] += 1
    q *= index.idf
    q /= np.linalg.norm(q) + 1e-12
    scores = index.matrix @ q  # Kosinus-Ähnlichkeit zu allen Chunks auf einmal (0 = nichts gemeinsam, 1 = identisch)
    top = np.argsort(-scores)[:k]  # Indizes der k höchsten Scores
    return [{**index.chunks[i], "score": float(scores[i])} for i in top]


# --------------------------------------------------------------------------- llm

# Kursbezug: gleiche Idee wie der System-Prompt in M07_VectorDB_RAG/RAG02_app.py
# ("Answer the user's question using only this information. Say 'I don't know' ...").
# Die Seitenangaben machen die Antwort überprüfbar.
SYSTEM_PROMPT = (
    "Beantworte die Frage ausschließlich mit Hilfe des bereitgestellten Kontexts. "
    "Steht die Antwort nicht im Kontext, sag das ehrlich. "
    "Antworte in der Sprache der Frage und nenne die Seiten, z. B. [S. 2]."
)


def provider() -> str:
    # Wählt automatisch den Anbieter, je nachdem welcher API-Key in der .env steht.
    # Hinweis: M01_Einführung/25_model_chat_anthropic.py liest CLAUDE_API_KEY – hier wird
    # ANTHROPIC_API_KEY erwartet (der Standardname, den ChatAnthropic automatisch liest).
    forced = os.getenv("RAG_PROVIDER", "").strip().lower()
    if os.getenv("RAG_MOCK") == "1" or forced == "mock":
        return "mock"
    if forced in ("openrouter", "anthropic"):
        return forced
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def _get_llm():
    # Kursbezug: dieselben Modellklassen wie in M01_Einführung –
    # ChatOpenRouter (chat_openrouter.py) und ChatAnthropic (25_model_chat_anthropic.py).
    # temperature=0 für möglichst gleichbleibende Antworten (wie in 40_simple_chain.py).
    # Das Modell wird erst beim ersten Aufruf erzeugt und dann wiederverwendet.
    global _llm
    with _lock:
        if _llm is None:
            p = provider()
            if p == "openrouter":
                from langchain_openrouter import ChatOpenRouter

                model = os.getenv("RAG_OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
                _llm = ChatOpenRouter(model=model, temperature=0)
            elif p == "anthropic":
                from langchain_anthropic import ChatAnthropic

                model = os.getenv("RAG_ANTHROPIC_MODEL", "claude-haiku-5-5")
                _llm = ChatAnthropic(model=model, temperature=0)
            else:
                raise RuntimeError("Kein LLM-Provider konfiguriert")
        return _llm


def _messages(question: str, sources: list[dict], history: list[dict]):
    # 6. Augmentation: Die gefundenen Chunks werden als "Kontext" in den Prompt eingefügt.
    # Kursbezug: dieselbe Liste aus ("system", ...)/("human", ...)-Tupeln wie bei
    # ChatPromptTemplate.from_messages(messages) in M01_Einführung/chat_openrouter.py und
    # RAG02_app.py. Statt Platzhaltern wie {query} wird der Text hier direkt per f-String
    # eingesetzt – das Ergebnis ist dasselbe wie prompt.invoke({...}).
    context = "\n\n".join(f"[S. {s['page']}] {s['text']}" for s in sources)
    msgs = [("system", SYSTEM_PROMPT)]
    # Gedächtnis: Die letzten 6 Nachrichten (= 3 Frage/Antwort-Paare) werden mitgeschickt,
    # damit Rückfragen wie "Erklär das genauer" funktionieren. Kursbezug: In
    # M09_Agents/langgraph_chat_summarization.py wird ab mehr als 6 Nachrichten
    # zusammengefasst – hier werden ältere Nachrichten einfach weggelassen.
    for m in history[-6:]:
        msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
    msgs.append(("human", f"Kontext:\n{context}\n\nFrage: {question}"))
    return msgs


def _text(content) -> str:
    # Kursbezug: entspricht res.content aus M01_Einführung. Manche Modelle liefern content
    # als Liste von Blöcken statt als String – diese Funktion macht daraus immer einen String
    # (ähnlich wie StrOutputParser in 40_simple_chain.py).
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _mock_answer(question: str, sources: list[dict]) -> str:
    # Ohne API-Key: Statt einer LLM-Antwort wird der beste gefundene Chunk zitiert.
    # So lässt sich das Retrieval auch ganz ohne LLM vorführen.
    best = sources[0] if sources else {"page": "?", "text": ""}
    return (
        f"(MOCK-Antwort) Zur Frage „{question}“ steht auf Seite {best['page']} u. a.: "
        f"„{best['text'][:160]}…“"
    )


def _complete(question: str, sources: list[dict], history: list[dict]) -> str:
    # 7. Generation. Kursbezug: model.invoke(...) wie in M01_Einführung/10_model_chat_openrouter.py.
    if provider() == "mock":
        time.sleep(float(os.getenv("RAG_MOCK_DELAY", "3")))
        return _mock_answer(question, sources)
    return _text(_get_llm().invoke(_messages(question, sources, history)).content)


def _stream(question: str, sources: list[dict], history: list[dict]) -> Iterator[str]:
    # Neu gegenüber scripts/: .stream() statt .invoke(). Jedes LangChain-Modell (und jede
    # Chain) kann beides – .stream() liefert die Antwort Stück für Stück, sobald das Modell
    # sie erzeugt. In Streamlit z. B. mit st.write_stream(...) anzeigen.
    if provider() == "mock":
        words = _mock_answer(question, sources).split(" ")
        delay = float(os.getenv("RAG_MOCK_DELAY", "3")) / max(len(words), 1)
        for w in words:
            time.sleep(delay)
            yield w + " "
        return
    for chunk in _get_llm().stream(_messages(question, sources, history)):
        yield _text(chunk.content)


# --------------------------------------------------------------------------- assistant

class Assistant:
    """Chat-Assistent zum PDF, der sich den Gesprächsverlauf merkt.

    Assistant(index=None): Ohne index wird load_index() aufgerufen.

    Attribute:
        history       bisheriger Verlauf: [{"role": "user" | "assistant", "content": str}, ...]
        last_sources  Quellen der letzten Antwort: [{"text", "page", "score"}, ...]
    """

    # Kursbezug: bündelt alle Schritte wie die Funktion rag() in M07_VectorDB_RAG/RAG02_app.py.
    # Die App ruft nur ask() auf – wie app.py die Funktion aus helper_funs.py in
    # M10_Deployment/our_web_app/. Anders als dort merkt sich das Objekt den Verlauf
    # (vgl. MemorySaver in M09_Agents/langchain_agent.py).

    def __init__(self, index: Index | None = None):
        self.index = index if index is not None else load_index()
        self.history: list[dict] = []
        self.last_sources: list[dict] = []

    def ask(self, question: str, k: int = 3) -> str:
        """Beantwortet eine Frage mit k Quellen und ergänzt den Verlauf."""
        # Retrieval -> Augmentation + Generation. last_sources entspricht den raw_docs in
        # RAG02_app.py, die dort unter "Retrieval" angezeigt werden.
        self.last_sources = retrieve(self.index, question, k)
        answer = _complete(question, self.last_sources, self.history)
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return answer

    def ask_stream(self, question: str, k: int = 3) -> Iterator[str]:
        """Wie ask(), liefert die Antwort aber stückweise (Generator von Strings).

        Der Verlauf wird ergänzt, sobald die Antwort vollständig ist.
        """
        self.last_sources = retrieve(self.index, question, k)
        sources, history = self.last_sources, list(self.history)

        def generate() -> Iterator[str]:
            parts = []
            for piece in _stream(question, sources, history):
                parts.append(piece)
                yield piece
            self.history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "".join(parts)},
            ]

        return generate()

    def summarize(self) -> str:
        """Fasst das Dokument zusammen (wird nicht in den Verlauf übernommen)."""
        # Kursbezug: Zusammenfassung per LLM statt mit einem vortrainierten
        # Summarization-Modell wie in M03_PretrainedNetworks/13_summarization.py.
        # Es werden die 5 passendsten Chunks als Kontext genutzt.
        return _complete(SUMMARY_PROMPT, retrieve(self.index, SUMMARY_PROMPT, 5), [])

    def reset(self) -> None:
        """Löscht den Verlauf."""
        self.history = []
        self.last_sources = []
