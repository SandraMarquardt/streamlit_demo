# PDF-Chat – Deployment-Ordner: DEMO Projekt

Zwei Streamlit-Apps zum RAG-Leitfaden, bereit für Streamlit Community Cloud:

| Datei | Was | Deployen mit |
| --- | --- | --- |
| `app.py` | Referenzlösung (besteht den Stresstest) | `OPENROUTER_API_KEY` (oder `ANTHROPIC_API_KEY`) |
| `naive_app.py` | Vergleichs-App mit typischen Agenten-Fehlern | **nur** `RAG_MOCK = "1"` |
| `rag_core.py` | RAG-Backend (Version 2, mit Zählern und Log-Zeilen) | – |
| `data/rag_leitfaden.pdf` | Das Dokument | – |

## Lokal starten

Aus diesem Ordner:

```bash
uv run streamlit run app.py
uv run streamlit run naive_app.py
```

Ohne uv: `pip install -r requirements.txt`, dann `streamlit run app.py`.

Den Key findet `rag_core` in der `.env` – in diesem Ordner oder einem darüber (z. B. im Projektordner). Ohne Key antwortet die App simuliert.

Sauberer Testlauf in einer frischen Umgebung (findet fehlende Pakete in `requirements.txt`):

```bash
uv run --isolated --no-project --with-requirements requirements.txt streamlit run app.py
```

## Deployment auf Streamlit Community Cloud

Pro App einmal **Create app** → Repository, Branch, Entrypoint:

1. `deployment/app.py` – unter **Advanced settings → Secrets**:

   ```toml
   OPENROUTER_API_KEY = "…"
   ```

2. `deployment/naive_app.py` – Secrets:

   ```toml
   RAG_MOCK = "1"
   ```

Bei beiden unter **Advanced settings** Python 3.12 wählen (damit sind die Versionen in `requirements.txt` getestet).

Ein Push auf den Branch deployt beide Apps neu.

## Deployment-Readiness-Review: Was auf dem Server anders ist

**Secrets**
- Auf dem Server gibt es keine `.env`. Keys kommen aus den App-Secrets. Secrets auf oberster Ebene stehen in Community Cloud auch als Umgebungsvariablen bereit; von dort liest `rag_core` sie. Das im Probelauf prüfen. Falls es nicht klappt, oben in beiden Apps ergänzen: `os.environ.update({k: str(v) for k, v in st.secrets.items()})`.
- `.env` und `.streamlit/secrets.toml` sind per `.gitignore` ausgeschlossen. Im Code steht kein Key (Suche nach `sk-` findet nichts).

**Kosten und Zugriff**
- Die URL ist öffentlich: Jeder, der sie kennt, verbraucht dein Guthaben. Einen Key mit Kreditlimit nehmen, die App in den Sharing-Einstellungen auf eingeladene Personen beschränken und den Key nach dem Workshop rotieren.
- `naive_app.py` nie mit echtem Key deployen: Alle Besucher teilen sich einen Chat-Verlauf, und „Neuer Chat“ löscht ihn für alle.

**Ein Prozess für alle Nutzer**
- Alle Besucher laufen im selben Python-Prozess. `st.cache_resource` (der Index) und `rag_core.stats()` gelten für den ganzen Server, `st.session_state` gilt pro Browser-Tab.
- Die Zähler beginnen nach jedem Neustart wieder bei 0.

**Neustarts, Ruhezustand, Dateisystem**
- Nach einer Weile ohne Besucher schläft die App ein. Beim Kaltstart wird der Index neu geladen (`RAG_LOAD_DELAY`, Standard 2 s).
- Was zur Laufzeit geschrieben wird, ist nach einem Neustart weg. Auch die Chat-Verläufe sind dann weg; nur „Chat herunterladen“ sichert sie.
- Das PDF muss im Repository liegen (`data/`). Ein Index, der nie committet wurde, existiert auf dem Server nicht.
- Ressourcen sind begrenzt. Für den TF-IDF-Index reicht das, für ein lokales Embedding-Modell nicht unbedingt.

**Abhängigkeiten**
- Community Cloud sucht die Paketdatei zuerst im Ordner des Entrypoints. Deshalb gilt `deployment/requirements.txt`, nicht `pyproject.toml`/`uv.lock` im Wurzelordner.
- `langchain-openrouter` und `langchain-anthropic` lädt `rag_core` erst, wenn ein Provider gewählt ist. Beide stehen in `requirements.txt`, damit beide Keys funktionieren.

**Logs**
- Die Zeilen `[rag_core] INFO llm call #…` erscheinen im Log der App (**Manage app**). Dort sieht man live, wer wie viel kostet.

