"""
build_features.py — build features.csv for the whole UD Hindi-HDTB train set
with our own pipeline, step by step, exactly as the web app's File mode does
it (webapp/pipeline.py: run_job).

    load .conllu -> filter -> generate variants -> word-order labels
    -> score with every scorer -> recommended word order -> features.csv

Output (reproducibility/data/, gitignored):
    features.csv            one row per (reference, variant) pair: the same
                            columns as the app's variants.csv — Sent_ID,
                            Variant_ID, Ref_Index, Variant_No, ML_Label,
                            Reference_Sentence, Variant_Sentence,
                            Reference_/Variant_Word_Order, every scorer's
                            *_Reference / *_Variant / Delta_*, Recommended_*
    passed_sentences.csv    the sentences that passed all 7 filters
    rejected_sentences.csv  the rest, with the reason
    summary.json            filter counts + word-order distribution

ML_Label alternates 1, 0, 1, 0 ... (generate_variants) and every Delta_* is
oriented by it (apply_scorers): reference - variant on 1-rows, variant -
reference on 0-rows — the paper's REF-VAR / VAR-REF coding.

Run from the repo root (takes ~65 min, the DSPS PCFG is the slow part):
    venv\\Scripts\\python.exe reproducibility\\build_features.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))                       # import the library from the repo root

# The PCFG parses ~88k sentences in one Java call; the app's 30-minute
# limit is too short for that, so allow 3 hours here.
os.environ.setdefault("HNT_PCFG_TIMEOUT", "10800")

import pandas as pd                                        # noqa: E402

from filtering import filter_sentences, summarize          # noqa: E402
from scoring import apply_scorers, build_corpus_context, get_scorers, recommend_column  # noqa: E402
from stanza_parser import load_input                       # noqa: E402
from variants import generate_variants, label_word_orders  # noqa: E402

CONLLU = REPO / "hi_hdtb-ud-train.conllu"
DATA_DIR = REPO / "reproducibility" / "data"
SCORES_DIR = DATA_DIR / "scores"    # one saved file per finished scorer (lets a rerun skip it)

# Same settings as a File-mode job with root_pos = "ud".
ROOT_POS = ["VERB", "AUX"]          # webapp/app.py ROOT_POS_PRESETS["ud"]
MIN_PHRASES = 2                     # fixed in the app
MAX_VARIANTS = 99                   # the paper's cap
SCHEME = "ud"

# Every built-in scorer except `surprisal` (Modelblocks / synproc, skipped).
# berkeley_pcfg is the app's DSPS PCFG (scoring/models/hdtb_dsps_grammar).
SCORERS = [
    "dependency_length",
    "information_status",
    "trigram",
    "lstm",
    "adaptive_lstm",
    "lstm_wiki50m",
    "adaptive_lstm_wiki50m",
    "berkeley_pcfg",
]
PCFG = "berkeley_pcfg"              # single-threaded Java: run it alongside the rest


def main():
    start = time.time()
    sentences = step1_load(CONLLU)
    passed, rejected = step2_filter(sentences)
    step3_save_filter_outputs(passed, rejected)
    pairs = step4_generate_variants(passed)
    pairs = step5_add_word_orders(pairs, passed)
    context = step6_build_context(sentences, passed)
    pairs = step7_score(pairs, context)
    pairs = step8_add_recommendation(pairs)
    step9_save(pairs)
    log(f"done in {(time.time() - start) / 60:.1f} min")


# ── steps ────────────────────────────────────────────────────────────────────

def step1_load(path):
    """Read the treebank. load_input stamps each sentence's position in the
    file (it becomes Ref_Index)."""
    log(f"1. loading {path.name}")
    sentences = load_input(str(path))
    log(f"   {len(sentences):,} sentences")
    return sentences


def step2_filter(sentences):
    """The 7 filters: questions, negation, empty nodes, non-projective, root
    not a finite verb, punctuation constituent, too few phrases / no subject
    or object. Also writes passed_ and rejected_sentences.csv."""
    log("2. filtering")
    passed, rejected, _passed_table = filter_sentences(
        sentences,
        allowed_root_pos=ROOT_POS,
        min_phrases=MIN_PHRASES,
        output_dir=str(DATA_DIR),
    )
    log(f"   {len(passed):,} passed, {len(rejected):,} rejected")
    return passed, rejected


def step3_save_filter_outputs(passed, rejected):
    """summary.json: rejection counts per filter and the SOV/DOSV/IOSV split."""
    summary = summarize(passed, rejected)
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    log(f"3. summary.json written — word orders {summary['word_order_distribution']}")


def step4_generate_variants(passed):
    """Permute the preverbal constituents. Only orders whose adjacent deprel
    pairs occur in the corpus are kept (File mode's grammar filter), at most
    99 per sentence. Adds the alternating ML_Label."""
    log("4. generating variants")
    pairs = generate_variants(passed, valid_deprel_pairs=None, max_variants=MAX_VARIANTS)
    log(f"   {len(pairs):,} pairs from {pairs['Ref_Index'].nunique():,} references")
    return pairs


def step5_add_word_orders(pairs, passed):
    """Reference_Word_Order / Variant_Word_Order: SOV, DOSV or IOSV."""
    log("5. adding word-order labels")
    return pairs.join(label_word_orders(pairs, passed))


def step6_build_context(sentences, passed):
    """What context-aware scorers need: the full corpus (to find a sentence's
    predecessor, incl. filtered-out ones), the parses, and the scheme."""
    log("6. building scorer context")
    return {"corpus": build_corpus_context(sentences), "passed": passed, "scheme": SCHEME}


def step7_score(pairs, context):
    """Run every scorer. The PCFG (one long Java process) runs in a second
    thread while the others run one after another; their new columns are
    then put together. Each apply_scorers call also writes that scorer's
    Delta_* column, oriented by ML_Label.

    Each scorer's new columns are saved to data/scores/ as soon as it
    finishes, and a scorer already saved there is not run again."""
    log(f"7. scoring with {len(SCORERS)} scorers")
    # Load the scorer registry once, before any thread starts: the first
    # get_scorers() call imports the scorer modules and is not thread-safe.
    get_scorers()
    others = [s for s in SCORERS if s != PCFG]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pcfg_job = pool.submit(score_with, pairs, PCFG, context)
        other_columns = [score_with(pairs, name, context) for name in others]
        pcfg_columns = pcfg_job.result()

    scored = pairs.copy()
    for columns in other_columns + [pcfg_columns]:
        for column in columns.columns:
            scored[column] = columns[column]
    return scored


def step8_add_recommendation(pairs):
    """Recommended_Sentence / _Is_Reference / _Score: the most natural order
    per reference by the scorers' weighted z-scores (as in variants.csv)."""
    log("8. adding recommended word order")
    recommended = recommend_column(pairs, SCORERS)
    return pairs.join(recommended) if recommended is not None else pairs


def step9_save(pairs):
    """Sort back into file order and write features.csv."""
    pairs = pairs.sort_values(["Ref_Index", "Variant_No"], kind="stable")
    path = DATA_DIR / "features.csv"
    pairs.to_csv(path, index=False, encoding="utf-8")
    log(f"9. wrote {path} — {len(pairs):,} rows, {len(pairs.columns)} columns")


# ── helpers ──────────────────────────────────────────────────────────────────

def score_with(pairs, name, context):
    """Run one scorer and return only the columns it added (incl. its Delta_*).
    Reuses data/scores/<name>.pkl when an earlier run already finished it."""
    saved = SCORES_DIR / f"{name}.pkl"
    if saved.exists():
        log(f"   {name:<24} reused {saved.name}")
        return pd.read_pickle(saved)
    started = time.time()
    result = apply_scorers(pairs, [name], context=context)
    new_columns = result[result.columns.difference(pairs.columns)]
    new_columns.to_pickle(saved)
    log(f"   {name:<24} {time.time() - started:7.1f} s  -> {', '.join(new_columns.columns)}")
    return new_columns


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    main()
