#!/usr/bin/env bash
#
# build.sh — compile the Taru C++ engine (synproc + ccmodel2synproccptmodel)
# for THIS machine. Run once after cloning, on any OS that has a C++17 compiler
# and the Armadillo + libxml2 libraries.
#
#   bash taru/build.sh
#
# The binaries are written to taru/workspace/bin/ and are git-ignored, because
# they are platform-specific — never commit them; everyone builds their own.
#
# Dependencies:
#   macOS (Homebrew):  brew install armadillo libxml2 openblas
#   Debian/Ubuntu:     sudo apt-get install -y g++ build-essential libarmadillo-dev libxml2-dev
#
set -euo pipefail

# Resolve the taru/ root (the dir this script lives in), regardless of CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$ROOT/workspace"
SRC="$ROOT/resource-incrsem/src"
mkdir -p "$WS/bin"

# Compiler + flags
CXX="${CXX:-g++}"
CFLAGS="$(cat "$ROOT/config/user-cflags.txt" 2>/dev/null || echo '-O3 -DNDEBUG')"

# Locate Armadillo / libxml2 includes (Homebrew vs system paths)
EXTRA_INC=""
EXTRA_LIB=""
if command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix)"
  for p in armadillo libxml2 openblas; do
    [ -d "$BREW_PREFIX/opt/$p/include" ] && EXTRA_INC="$EXTRA_INC -I$BREW_PREFIX/opt/$p/include"
    [ -d "$BREW_PREFIX/opt/$p/lib" ]     && EXTRA_LIB="$EXTRA_LIB -L$BREW_PREFIX/opt/$p/lib"
  done
  EXTRA_INC="$EXTRA_INC -I$BREW_PREFIX/include -I$BREW_PREFIX/include/libxml2"
  EXTRA_LIB="$EXTRA_LIB -L$BREW_PREFIX/lib"
fi
[ -d /usr/include/libxml2 ] && EXTRA_INC="$EXTRA_INC -I/usr/include/libxml2"

echo "Compiler: $CXX"
echo "Flags:    $CFLAGS"

for NAME in synproc ccmodel2synproccptmodel; do
  echo "Building $NAME ..."
  $CXX \
    -I"$SRC" \
    -I"$ROOT/resource-incrsem/include" \
    -I"$ROOT/resource-logreg/include" \
    -I"$ROOT/resource-rvtl" \
    $EXTRA_INC \
    -Wall $CFLAGS -fpermissive -std=c++17 \
    "$SRC/$NAME.cpp" \
    $EXTRA_LIB -lm -larmadillo -lpthread \
    -o "$WS/bin/$NAME"
  chmod +x "$WS/bin/$NAME"
  echo "  -> $WS/bin/$NAME  OK"
done

echo ""
echo "Done. Binaries are in taru/workspace/bin/"
echo "Quick check:"
ls -la "$WS/bin/"
