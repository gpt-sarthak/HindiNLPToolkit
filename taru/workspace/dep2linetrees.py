#!/usr/bin/env python3
"""
dep2linetrees.py

Convert a CoNLL-U dependency file (HDTB format) into head-first constituency
trees in .linetrees format (one tree per line, no sent_id).

The ONLY argument is the input .txt file. The output is written automatically
to  constituencytree_<timestamp>.linetrees  in the current directory.

  * Yadav (2017)-style head-first projection (head words first, then children).
  * XPOS (col 5, falling back to UPOS) as preterminal tags.
  * Works whether or not the input has `# sent_id` / comment lines.

Usage:
    python dep2linetrees.py input.txt
"""

import re
import sys
from datetime import datetime
from collections import OrderedDict, defaultdict


def read_conllu(path):
    """Return a list of sentences, each a list of token dicts (from a file path)."""
    with open(path, encoding="utf-8") as f:
        return _read_conllu_lines(f)


def read_conllu_text(text):
    """Same as read_conllu but from an in-memory string (for the web API)."""
    return _read_conllu_lines(text.splitlines())


def _read_conllu_lines(lines):
    sents, cur = [], []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith("#"):          # ignore sent_id / text / any comment
            continue
        cols = line.split("\t")
        if len(cols) < 10 or "-" in cols[0] or "." in cols[0]:
            continue
        try:
            cid, ctype = "none", "head"
            for feat in cols[9].split("|"):
                if feat.startswith("ChunkId="):
                    cid = feat.split("=", 1)[1] or "none"
                elif feat.startswith("ChunkType="):
                    ctype = feat.split("=", 1)[1] or "head"
            cur.append({
                "id": int(cols[0]), "word": cols[1],
                "upos": cols[3], "xpos": cols[4],
                "head": int(cols[6]), "deprel": cols[7],
                "chunk_id": cid, "chunk_type": ctype,
            })
        except (ValueError, IndexError):
            continue
    if cur:
        sents.append(cur)
    return sents


def conllu_to_trees(text):
    """CoNLL-U text -> list of (S ...) linetree strings. Importable entry point."""
    return [t for t in (convert(s) for s in read_conllu_text(text)) if t]


def group_into_chunks(tokens):
    chunks = OrderedDict()
    for t in tokens:
        cid = t["chunk_id"]
        if cid not in chunks:
            chunks[cid] = {"label": re.sub(r"\d+$", "", cid) or cid,
                           "tokens": [], "head": None}
        chunks[cid]["tokens"].append(t)
        if t["chunk_type"] == "head" and chunks[cid]["head"] is None:
            chunks[cid]["head"] = t
    for c in chunks.values():
        if c["head"] is None:
            c["head"] = c["tokens"][0]
    return chunks


def build_chunk_deps(chunks):
    tok2chunk = {t["id"]: cid for cid, c in chunks.items() for t in c["tokens"]}
    deps, root = defaultdict(list), None
    for cid, c in chunks.items():
        parent = c["head"]["head"]
        if parent == 0:
            root = cid
        elif parent in tok2chunk and tok2chunk[parent] != cid:
            deps[tok2chunk[parent]].append(cid)
    return deps, root


def build_phrase(cid, chunks, deps, visited):
    if cid in visited:
        return ""
    visited.add(cid)
    c = chunks[cid]
    terms = []
    for t in c["tokens"]:
        pos = t["xpos"] if t["xpos"] != "_" else t["upos"]
        terms.append(f"({pos} {t['word']})")
    content = " ".join(terms)                       # head words FIRST
    kids = [build_phrase(ch, chunks, deps, visited) for ch in deps.get(cid, [])]
    kids = [k for k in kids if k]
    if kids:
        content = content + " " + " ".join(kids)    # children appended AFTER
    return f"({c['label']} {content})"


def convert(tokens):
    if not tokens:
        return ""
    chunks = group_into_chunks(tokens)
    deps, root = build_chunk_deps(chunks)
    if not root:
        root = next(iter(chunks))
    tree = build_phrase(root, chunks, deps, set())
    if not tree.startswith("(S "):
        tree = f"(S {tree})"
    return tree


def main():
    if len(sys.argv) != 2:
        print("usage: python dep2linetrees.py input.txt")
        sys.exit(1)

    sentences = read_conllu(sys.argv[1])
    trees = [t for t in (convert(s) for s in sentences) if t]

    out_path = f"constituencytree_{datetime.now():%Y%m%d_%H%M%S}.linetrees"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trees) + "\n")

    print(f"Wrote {len(trees)} trees -> {out_path}")


if __name__ == "__main__":
    main()
