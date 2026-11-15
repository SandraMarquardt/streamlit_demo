# Day 5 starter (trainer notes)

Runs entirely in the course `.venv`: no new venv, no new packages (only streamlit, numpy, pypdf,
python-dotenv, langchain-openrouter / langchain-anthropic, all in `uv.lock`).

```
tag5_starter/
├── ANLEITUNG.md          # participant handout (German). Stays OUTSIDE the app folder
├── tag5_einrichten.py    # copies the kit to unsere_skripte/tag5_app/ (flags: --gruppe-b, --messung, --fallback, --loesung)
├── kit/                  # what every participant gets at the start
│   ├── rag_core.py       # VERSION 1: Assistant(index=None) with memory, no counters, no log lines
│   ├── opencode.json     # pins the Open Code model: openrouter/anthropic/claude-haiku-5.5
│   ├── check_setup.py    # green/red pre-flight check
│   └── AGENTS.md         # environment facts + neutral API description, no framework rules
├── messung/rag_core.py   # VERSION 2: same API + stats() + "[rag_core] INFO ..." lines. Installed with --messung
├── gruppe_b/             # Anfrage_Weiterbildung.md (stakeholder brief), PRD_vorlage.md, wireframe.png
├── fallback/naive_app.py # comparison app (caches the whole Assistant). --fallback, installs version 2
├── loesung/fixed_app.py  # reference solution. --loesung after the rebuild round, installs version 2
└── trainer/              # stresstest.py (AppTest), interrupt_test.py (check 5, Playwright), abnahme_checkliste.md
```

The PDF is not duplicated: the setup script copies
`scripts/M07_VectorDB_RAG/data/Retrieval Augmented Generation.pdf` to `tag5_app/data/rag_leitfaden.pdf`.

## Participant commands (from the repo root, `.venv` active)

```bash
git pull
python scripts/M10_Deployment/tag5_starter/tag5_einrichten.py
python unsere_skripte/tag5_app/check_setup.py
cd unsere_skripte/tag5_app && opencode          # agent works in this folder; once: /connect → OpenRouter → key
python scripts/M10_Deployment/tag5_starter/tag5_einrichten.py --messung   # stage 2: rag_core version 2
streamlit run unsere_skripte/tag5_app/app.py    # from the repo root
```

No API key, or the network is down? Put `TAG5_MOCK=1` in `.env`. Every stress-test check still works.

## Verified results (mock mode, streamlit 1.60.0, rag_core version 2)

| | K1 | K2 | K3 | K4 | B1 | B2 | B3 | B4 | key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_app | ✗ | n/a | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| fixed_app | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Numbering = `ANLEITUNG.md` (K = core checks, B = bonus; the key search belongs to the deployment step).
K2 (interrupt) is tested with `trainer/interrupt_test.py` in a real browser; the naive app uses `text_input`, which that script
doesn't cover. K4 must be tested with two sessions open at the same time - a reload starts a new session and hides the bug.

```bash
python scripts/M10_Deployment/tag5_starter/trainer/stresstest.py unsere_skripte/tag5_app/naive_app.py
```

## Distribution note

`git pull` delivers everything to participants. `loesung/` is the solution: commit it after the session,
or keep it out of the repo until then.
