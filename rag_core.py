"""rag_core (Version 2, mit Messpunkten) – RAG-Backend für Tag 5: Fragen zu einem PDF beantworten.

Neu in Version 2: stats() und INFO-Zeilen im Terminal. Sonst identisch zu Version 1.

Nicht ändern – nur importieren.

    import rag_core
    assistant = rag_core.Assistant()
    print(assistant.ask("Was ist RAG?"))

Retrieval per TF-IDF (numpy), LLM über OpenRouter oder Anthropic (Key aus der .env im Repo-Root).
Das Modul schreibt INFO-Zeilen ins Terminal.

Umgebungsvariablen (optional, z. B. in der .env):
    TAG5_MOCK=1              Antworten simulieren statt LLM aufzurufen
    TAG5_PROVIDER            openrouter | anthropic | mock   (Standard: automatisch)
    TAG5_OPENROUTER_MODEL    Standard: z-ai/glm-5.3-flash
    TAG5_ANTHROPIC_MODEL     Standard: claude-haiku-5-5
    TAG5_LOAD_DELAY          Ladezeit eines Embedding-Modells simulieren, in Sekunden (Standard 2)
    TAG5_MOCK_DELAY          Antwortzeit im Mock-Modus, in Sekunden (Standard 3)
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
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUMMARY_PROMPT = "Fasse das Dokument in 5 Stichpunkten zusammen."


def _load_env() -> Path | None:
    """Sucht die .env von diesem Ordner aufwärts (also auch die im Repo-Root)."""
    for folder in [HERE, *HERE.parents]:
        env_file = folder / ".env"
        if env_file.is_file():
            from dotenv import load_dotenv

            load_dotenv(env_file)  # überschreibt keine bereits gesetzten Variablen
            return env_file
    return None


ENV_FILE = _load_env()

_lock = threading.Lock()
_stats = {"index_loads": 0, "llm_calls": 0}
_llm = None


def _log(msg: str) -> None:
    print(f"[rag_core] INFO {msg}", flush=True)


def stats() -> dict:
    """Zähler: index_loads, llm_calls."""
    with _lock:
        return dict(_stats)


# --------------------------------------------------------------------------- index

@dataclass
class Index:
    name: str
    chunks: list[dict]  # {"text": str, "page": int}
    vocab: dict[str, int]
    idf: np.ndarray
    matrix: np.ndarray  # L2-normalisierte TF-IDF-Zeilen


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


def _chunk(pages: list[tuple[int, str]]) -> list[dict]:
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
    with _lock:
        _stats["index_loads"] += 1
        n = _stats["index_loads"]
    _log(f"index loaded #{n} ({name})")
    time.sleep(float(os.getenv("TAG5_LOAD_DELAY", "2")))  # simuliert Embedding-Modell laden

    reader = PdfReader(pdf_source)
    pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    chunks = _chunk(pages)
    if not chunks:
        raise ValueError(f"Kein Text im PDF gefunden: {name}")

    docs = [_tokens(c["text"]) for c in chunks]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    tf = np.zeros((len(docs), len(vocab)))
    for row, doc in enumerate(docs):
        for t in doc:
            tf[row, vocab[t]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    matrix = tf * idf
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return Index(name=name, chunks=chunks, vocab=vocab, idf=idf, matrix=matrix)


def load_index(pdf_path: str | Path | None = None) -> Index:
    """Liest ein PDF ein und baut den Suchindex (Standard: data/rag_leitfaden.pdf)."""
    path = Path(pdf_path) if pdf_path else DEFAULT_PDF
    return _build(str(path), path.name)


def index_from_bytes(data: bytes, name: str = "upload.pdf") -> Index:
    """Wie load_index(), für ein hochgeladenes PDF (Bytes)."""
    return _build(BytesIO(data), name)


def retrieve(index: Index, question: str, k: int = 3) -> list[dict]:
    """Die k passendsten Abschnitte: [{"text", "page", "score"}, ...]."""
    q = np.zeros(len(index.vocab))
    for t in _tokens(question):
        if t in index.vocab:
            q[index.vocab[t]] += 1
    q *= index.idf
    q /= np.linalg.norm(q) + 1e-12
    scores = index.matrix @ q
    top = np.argsort(-scores)[:k]
    return [{**index.chunks[i], "score": float(scores[i])} for i in top]


# --------------------------------------------------------------------------- llm

SYSTEM_PROMPT = (
    "Beantworte die Frage ausschließlich mit Hilfe des bereitgestellten Kontexts. "
    "Steht die Antwort nicht im Kontext, sag das ehrlich. "
    "Antworte in der Sprache der Frage und nenne die Seiten, z. B. [S. 2]."
)


def provider() -> str:
    forced = os.getenv("TAG5_PROVIDER", "").strip().lower()
    if os.getenv("TAG5_MOCK") == "1" or forced == "mock":
        return "mock"
    if forced in ("openrouter", "anthropic"):
        return forced
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def _get_llm():
    global _llm
    with _lock:
        if _llm is None:
            p = provider()
            if p == "openrouter":
                from langchain_openrouter import ChatOpenRouter

                model = os.getenv("TAG5_OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
                _llm = ChatOpenRouter(model=model, temperature=0)
            elif p == "anthropic":
                from langchain_anthropic import ChatAnthropic

                model = os.getenv("TAG5_ANTHROPIC_MODEL", "claude-haiku-5-5")
                _llm = ChatAnthropic(model=model, temperature=0)
            else:
                raise RuntimeError("Kein LLM-Provider konfiguriert")
        return _llm


def _messages(question: str, sources: list[dict], history: list[dict]):
    context = "\n\n".join(f"[S. {s['page']}] {s['text']}" for s in sources)
    msgs = [("system", SYSTEM_PROMPT)]
    for m in history[-6:]:
        msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
    msgs.append(("human", f"Kontext:\n{context}\n\nFrage: {question}"))
    return msgs


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _count_call(question: str) -> None:
    with _lock:
        _stats["llm_calls"] += 1
        n = _stats["llm_calls"]
    _log(f"llm call #{n} ({provider()}): {question[:60]!r}")


def _mock_answer(question: str, sources: list[dict]) -> str:
    best = sources[0] if sources else {"page": "?", "text": ""}
    return (
        f"(MOCK-Antwort) Zur Frage „{question}“ steht auf Seite {best['page']} u. a.: "
        f"„{best['text'][:160]}…“"
    )


def _complete(question: str, sources: list[dict], history: list[dict]) -> str:
    _count_call(question)
    if provider() == "mock":
        time.sleep(float(os.getenv("TAG5_MOCK_DELAY", "3")))
        return _mock_answer(question, sources)
    return _text(_get_llm().invoke(_messages(question, sources, history)).content)


def _stream(question: str, sources: list[dict], history: list[dict]) -> Iterator[str]:
    _count_call(question)
    if provider() == "mock":
        words = _mock_answer(question, sources).split(" ")
        delay = float(os.getenv("TAG5_MOCK_DELAY", "3")) / max(len(words), 1)
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

    def __init__(self, index: Index | None = None):
        self.index = index if index is not None else load_index()
        self.history: list[dict] = []
        self.last_sources: list[dict] = []

    def ask(self, question: str, k: int = 3) -> str:
        """Beantwortet eine Frage mit k Quellen und ergänzt den Verlauf."""
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
        return _complete(SUMMARY_PROMPT, retrieve(self.index, SUMMARY_PROMPT, 5), [])

    def reset(self) -> None:
        """Löscht den Verlauf."""
        self.history = []
        self.last_sources = []
