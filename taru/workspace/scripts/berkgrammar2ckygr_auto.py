#!/usr/bin/env python3
"""
berkgrammar2ckygr_auto.py
Berkeley grammar -> CKY grammar, with AUTOMATIC root-label detection.

Generalizes the S_0 fix: instead of hardcoding ROOT_0 (English) or S_0 (Hindi),
we detect the root label(s) as parents that never appear as a child. Works for
any treebank (Hindi, Tamil, Bengali, custom corpora...).

Usage: cat model.grammar | sed 's/[+]/-/g' | python3 berkgrammar2ckygr_auto.py > out.x-ccu.model
"""
import sys

lines = [ln.rstrip("\n") for ln in sys.stdin]

# pass 1: find root label(s) = parents that never occur as a child
parents, children = set(), set()
for ln in lines:
    arr = ln.split()
    if len(arr) < 4:
        continue
    parents.add(arr[0])
    if len(arr) >= 4:
        children.add(arr[2])
    if len(arr) >= 5:
        children.add(arr[3])

roots = {p for p in parents if p not in children}
if not roots:  # fallback to the known conventions
    roots = {r for r in ("ROOT_0", "S_0", "TOP_0") if r in parents}
sys.stderr.write("[berkgrammar2ckygr_auto] root label(s): %s\n" % (sorted(roots) or "NONE FOUND"))

# pass 2: emit rules
for ln in lines:
    arr = ln.split()
    if len(arr) < 4:
        print(ln, end="")
        continue
    if arr[0] in roots:                 # unary root rule -> Cr
        if arr[2] not in roots:
            print("Cr : %s = %s" % (arr[2], arr[3]), end="")
    elif len(arr) >= 5:                 # binary rule -> R
        print("R : %s %s %s" % (arr[0], arr[2], arr[3]), end="")
