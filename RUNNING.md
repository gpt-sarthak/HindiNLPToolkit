# Running the toolkit (with the Taru Tree & Surprisal tool)

The Taru engine includes two C++ programs (`synproc`, `ccmodel2synproccptmodel`)
that must be **compiled for the machine they run on**. The repo does *not* ship
prebuilt binaries — they are platform-specific (a macOS binary won't run on
Linux and vice-versa). Pick one of the two paths below.

## Option A — Docker (one command, identical everywhere)

Recommended if you just want it to work. Everything (Java, the C++ compiler,
the libraries, the binary build) happens inside a Linux image, so it runs the
same on macOS, Windows, and Linux.

```bash
docker build -t hindi-nlp-toolkit .
docker run --rm -p 8001:8001 hindi-nlp-toolkit
```

Then open:
- http://localhost:8001/        — Word-Order pipeline
- http://localhost:8001/taru    — Tree & Surprisal tool

## Option B — Native (run directly, no Docker)

Requires a C++17 compiler, Java, and a few libraries.

```bash
# 1. system libraries for the C++ engine
#    macOS:          brew install armadillo libxml2 openblas
#    Debian/Ubuntu:  sudo apt-get install -y g++ build-essential libarmadillo-dev libxml2-dev default-jre-headless

# 2. compile the engine for THIS machine (writes taru/workspace/bin/)
bash taru/build.sh

# 3. python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. run
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8001
```

Same URLs as above.

## Why no prebuilt binaries?

`synproc` and `ccmodel2synproccptmodel` are compiled C++ that links against
Armadillo and libxml2. A binary built on Apple Silicon won't execute on a Linux
server (different CPU + libraries). Shipping source + a build step is how the
original Hugging Face Space worked too: it recompiled `synproc` on Linux at
image-build time. Both `taru/build.sh` and the `Dockerfile` use the same
compile recipe, so wherever you build, you get a binary native to that machine.
