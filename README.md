---
title: Hindi NLP Toolkit
emoji: 📝
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8001
pinned: false
---

# HindiNLPToolKit

Python library + web app for **dependency-length-minimization (DLM)** research
on Hindi: filter CoNLL-U sentences, permute preverbal constituents, build ML
training pairs (reference vs. variant order), score them, and evaluate how well
each scorer predicts the attested word order.

## What it does

- **Filter** raw or annotated Hindi sentences down to clean, permutable clauses.
- **Generate variants** by reordering preverbal constituents (surface pairs).
- **Score** each pair with 9 plugin scorers: dependency length, information
  status, incremental constituency **surprisal** (Taru), **trigram**, **LSTM**,
  **adaptive LSTM**, a Berkeley **PCFG** log-likelihood, and the retrained
  **LSTM (Wikipedia 50M)** pair.
- **Evaluate** pairwise ranking accuracy (File mode) or **recommend** the most
  natural word order for a single typed sentence (Sentence mode).

## Running

This Space builds the `Dockerfile`, which installs Java + a C++ toolchain,
compiles the Taru `synproc` engine for Linux, and serves the FastAPI app on
port **8001**. See `RUNNING.md` for local Docker / native instructions and
`DOCS.md` for the full API reference.

The app opens on the word-order pipeline; `/taru` is the Tree & Surprisal tool
and `/how-it-works` explains the pipeline end to end.
