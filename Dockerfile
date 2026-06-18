# Dockerfile — run the full HindiNLPToolkit (with the embedded Taru Tree &
# Surprisal tool) in one Linux container. Works the same on Mac/Windows/Linux
# because everything happens inside the image.
#
#   docker build -t hindi-nlp-toolkit .
#   docker run --rm -p 8001:8001 hindi-nlp-toolkit
#
# Then open http://localhost:8001/  (pipeline) and http://localhost:8001/taru
#
# The synproc / ccmodel2synproccptmodel binaries are COMPILED here for Linux —
# the repo never ships prebuilt binaries (they are platform-specific).

FROM python:3.11-slim

# --- system deps: Java (Berkeley parser), C++ toolchain, Armadillo, libxml2 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
        g++ build-essential \
        libarmadillo-dev libxml2-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# --- compile the Taru C++ engine for Linux ---
RUN ROOT=/app/taru && \
    SRC=$ROOT/resource-incrsem/src && \
    mkdir -p $ROOT/workspace/bin && \
    CFLAGS="$(cat $ROOT/config/user-cflags.txt 2>/dev/null || echo '-O3 -DNDEBUG')" && \
    for NAME in synproc ccmodel2synproccptmodel; do \
        g++ -I$SRC \
            -I$ROOT/resource-incrsem/include \
            -I$ROOT/resource-logreg/include \
            -I$ROOT/resource-rvtl \
            -I/usr/include/libxml2 \
            -Wall $CFLAGS -fpermissive -std=c++17 \
            $SRC/$NAME.cpp \
            -lm -larmadillo -lpthread \
            -o $ROOT/workspace/bin/$NAME && \
        chmod +x $ROOT/workspace/bin/$NAME && \
        echo "$NAME built OK"; \
    done

# --- python deps ---
RUN pip install --no-cache-dir -r requirements.txt

# writable scratch dirs used at runtime
RUN mkdir -p taru/workspace/results taru/workspace/genmodel/custom && \
    chmod -R 777 taru/workspace/results taru/workspace/genmodel/custom

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8001"]
