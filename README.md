# Hindi NLP Toolkit — Word-Order Pipeline + Tree & Surprisal

A web toolkit for Hindi word-order research. It filters a corpus, generates
preverbal constituent-order variants, and scores (reference, variant) pairs with
pluggable metrics (dependency length, information status, **constituency
surprisal**). It also bundles **तरु (Taru)**, a Tree & Surprisal tool: parse any
Hindi sentence into a constituency tree with per-word ModelBlocks surprisal, and
train custom models from your own corpus.

Everything runs as one FastAPI app:

- `/` — the Word-Order pipeline (upload -> parse -> filter -> variants -> download)
- `/taru` — the Tree & Surprisal tool (parse / train, sentences or CoNLL-U)
- `/docs` — interactive API docs

---

## Quick start

The Taru engine includes two C++ programs (`synproc`,
`ccmodel2synproccptmodel`) that **must be compiled for the machine they run
on** — the repo ships source + a build recipe, not prebuilt binaries. Pick one
of the two options.

### Option A — Docker (one command, works on any OS)

Recommended. Java, the C++ toolchain, the libraries, and the binary build all
happen inside a Linux image, so it behaves identically on macOS, Windows, and
Linux.

```bash
docker build -t hindi-nlp-toolkit .
docker run --rm -p 8001:8001 hindi-nlp-toolkit
```

Open http://localhost:8001/ and http://localhost:8001/taru

### Option B — Native (no Docker)

Needs a C++17 compiler, Java, and a couple of libraries.

```bash
# 1. system libraries for the C++ engine
#    macOS:          brew install armadillo libxml2 openblas
#    Debian/Ubuntu:  sudo apt-get install -y g++ build-essential libarmadillo-dev libxml2-dev default-jre-headless

# 2. compile the engine for THIS machine (writes taru/workspace/bin/)
bash taru/build.sh

# 3. python deps (Python 3.9+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. run
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8001
```

Open http://localhost:8001/ and http://localhost:8001/taru

> Requires Git LFS to fetch the bundled model/jar. If you cloned without it:
> `git lfs install && git lfs pull`.

---

## Using it

**Word-Order pipeline (`/`)** — drop a `.conllu` treebank or a `.txt` of Hindi
sentences (one per line, parsed with Stanza), choose options and scorers, run.
Each stage's output is downloadable as it finishes. Tick the **`surprisal`**
scorer to add a `Delta_Surprisal` column computed by the bundled Taru engine.

**Tree & Surprisal tool (`/taru`)**
- *Parse / Test* — enter Hindi sentence(s) **or** CoNLL-U dependency trees;
  get the constituency tree + per-word surprisal. Choose the model (built-in
  HDTB or one you trained) and toggle surprisal on/off.
- *Train a model* — upload raw sentences (-> silver trees via the base parser)
  or CoNLL-U dependency trees (-> gold trees), train a fresh grammar + surprisal
  model, and download the model + generated training set.

---

## Project layout

```
webapp/            FastAPI app (routes only)
  app.py           pipeline routes + mounts the Taru router
  taru_routes.py   /taru page + /taru/parse, /train, /models, /download
  static/index.html
scoring/           pluggable pair scorers (drop a Scorer subclass to add one)
  surprisal_scorer.py   constituency surprisal via the Taru engine
filtering/ variants/ helpers/ stanza_parser/   pipeline logic
taru/              the Tree & Surprisal engine
  workspace/       backend, UI, training pipeline, models (LFS)
  resource-*/      ModelBlocks C++ sources compiled by build.sh / Dockerfile
  build.sh         compile the engine for the current machine
Dockerfile         build + run the whole toolkit in one container
```

---

## Notes

- **No prebuilt binaries.** `synproc` / `ccmodel2synproccptmodel` are compiled
  per machine (a macOS binary won't run on Linux). Both `taru/build.sh` and the
  Dockerfile use the same compile recipe.
- **Custom models** trained via `/taru` are saved under
  `taru/workspace/genmodel/custom/` and persist across restarts when run
  natively (the container's are ephemeral unless you mount a volume).
- **Adding a scorer:** create a file in `scoring/` subclassing
  `scoring.base.Scorer` — it's auto-discovered and shows up as a checkbox. See
  `scoring/example_scorer.py` and `CONTRIBUTING.md`.
