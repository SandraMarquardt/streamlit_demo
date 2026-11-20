# Chat mit dem RAG-Leitfaden

Streamlit-App, die Fragen zum RAG-Leitfaden (`data/rag_leitfaden.pdf`) beantwortet. Antworten stammen nur aus dem Leitfaden und zeigen die Quellen (Seite und Ausschnitt).

## Dateien

- `app.py`: Streamlit-Oberfläche
- `rag_core.py`: Backend (Index, Retrieval, LLM-Anbindung)
- `data/rag_leitfaden.pdf`: Dokument
- `requirements.txt`: Abhängigkeiten mit festen Versionen

## Lokal starten

Python 3.12 verwenden. Danach im Ordner `deployments/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Die App läuft dann unter http://localhost:8501.

## API-Key hinterlegen

Der Key wird nie in den Code geschrieben. Es gibt zwei Wege:

**Lokal:** Datei `.env` im Ordner `deployments/` anlegen:

```
OPENROUTER_API_KEY=dein-key
```

**Oder:** Datei `.streamlit/secrets.toml` anlegen:

```toml
OPENROUTER_API_KEY = "dein-key"
```

Beide Dateien stehen in `.gitignore` und landen nicht im Repository. Alternativ ist ein Anthropic-Key über `ANTHROPIC_API_KEY` möglich.

Ohne Key läuft die App im Mock-Modus, und die Antworten sind nur Platzhalter. Die App zeigt dann einen Hinweis.

## Auf einem Server starten

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Auf Streamlit Community Cloud: Repository verbinden, Hauptdatei `app.py`, Abhängigkeiten aus `requirements.txt`, und den Key unter **Settings → Secrets** eintragen.

## Hinweise

- Die App hat keine Anmeldung. Wer die URL kennt, kann den API-Key verbrauchen. Bitte den Zugang auf dem Server oder über den Hoster einschränken.
- Der Chat wird nicht gespeichert. Nach dem Neuladen der Seite ist er leer. Der Download ist die einzige dauerhafte Ausgabe.
