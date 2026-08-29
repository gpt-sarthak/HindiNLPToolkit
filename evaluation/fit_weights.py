"""
evaluation.fit_weights
======================
Offline: fit the recommender's feature weights on a corpus and write them to
``scoring/models/recommender_weights.json``.

The Sentence-mode recommender (``scoring.recommend``) combines each scorer's
feature via a weighted mean of direction-oriented z-scores.  The weights are
the *standardized univariate* logistic-regression coefficients from
``evaluation.evaluate_pairs`` — trained on (reference, variant) pairs, they
transfer to ranking single candidates (the Joachims pairwise trick).  Only the
coefficient *magnitude* is stored: the sign comes from each scorer's declared
``higher_is_more_natural``, so a noisy fit can never flip a theory-verified
direction (a sign disagreement is reported as a warning instead).

Usage (from the project root, venv active)::

    python -m evaluation.fit_weights
    python -m evaluation.fit_weights --corpus hi_hdtb-ud-train.conllu \
        --scheme ud --scorers dependency_length,information_status \
        --out scoring/models/recommender_weights.json

Defaults fit dependency_length + information_status on the committed UD HDTB
corpus (both deterministic — no model files needed).  Rerun with more scorers
once their features have been scored corpus-wide (trigram/lstm/pcfg are slow).

Note on reproducing the committed ``Delta_DL`` (0.6377, n=193474): that fit
predates the 2026-08-03 verb-only root change, when the ``ud`` preset here
still admitted NOUN/ADJ/PROPN roots.  Re-running now filters out HDTB's ~1,546
copular trees, so it draws on a smaller corpus and will land near but not on
the committed value.  The committed number is a documented historical fit, not
a target to reproduce exactly.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = "hi_hdtb-ud-train.conllu"
DEFAULT_SCHEME = "ud"
DEFAULT_SCORERS = "dependency_length,information_status"
DEFAULT_OUT = "scoring/models/recommender_weights.json"

# Same presets the webapp uses (webapp/app.py ROOT_POS_PRESETS); duplicated
# here so the library-side script does not import from webapp/.  Keep the two
# in step: they drifted once (this copy kept NOUN/ADJ/PROPN for two days after
# the app went verb-only), which would have refit the weights on a corpus the
# app no longer accepts.
ROOT_POS_PRESETS = {
    "paninian": None,  # library defaults: VERB / AUX / VM / VAUX
    "ud": ["VERB", "AUX"],
}


def fit_weights(corpus: str, scheme: str, scorer_names: list, out_path: Path) -> dict:
    """Run parse-free pipeline (filter → variants → scorers → evaluate) over
    *corpus* and write the weights JSON.  Returns the written payload."""
    from evaluation import evaluate_pairs
    from filtering import filter_sentences
    from scoring import apply_scorers, build_corpus_context, get_scorers
    from stanza_parser import load_input
    from variants import generate_variants

    print(f"Loading corpus: {corpus}")
    sentences = load_input(corpus)
    print(f"  {len(sentences)} sentences")

    passed, _rejected_df, _passed_df = filter_sentences(
        sentences, allowed_root_pos=ROOT_POS_PRESETS[scheme]
    )
    print(f"  {len(passed)} passed filters")

    pairs_df = generate_variants(passed)
    print(f"  {len(pairs_df)} (reference, variant) pairs")

    context = {
        "corpus": build_corpus_context(sentences),
        "passed": passed,
        "scheme": scheme,
    }
    pairs_df = apply_scorers(pairs_df, scorer_names, context=context)

    delta_cols = [c for c in pairs_df.columns if c.startswith("Delta_")]
    result = evaluate_pairs(pairs_df, deltas=delta_cols)

    # Direction sanity: delta = first − second with label 1 = reference-first,
    # so a negative coefficient means "reference preferred when lower" — which
    # must match higher_is_more_natural=False (and vice versa).
    declared = {
        delta: hib
        for scorer in get_scorers().values()
        # fall back to the pre-rename attribute for un-migrated scorers
        for delta, hib in (
            scorer.higher_is_more_natural or getattr(scorer, "higher_is_better", {})
        ).items()
    }
    weights, meta_predictors = {}, {}
    for p in result["predictors"]:
        coef = p["coefficient"]
        weights[p["delta"]] = abs(coef)
        meta_predictors[p["delta"]] = {
            "coefficient": coef,
            "accuracy": p["accuracy"],
            "n": p["n"],
        }
        expected = declared.get(p["delta"])
        if expected is not None and coef != 0 and (coef > 0) != expected:
            print(
                f"  WARNING: fitted coefficient for {p['delta']} is "
                f"{coef:+.4f}, but the scorer declares higher_is_more_natural="
                f"{expected} — the declared direction is kept; check the fit."
            )

    payload = {
        "weights": weights,
        "meta": {
            "predictors": meta_predictors,
            "corpus": corpus,
            "scheme": scheme,
            "scorers": scorer_names,
            "n_pairs": result["n_pairs"],
            "generated": date.today().isoformat(),
            "method": "abs(standardized univariate logistic coefficient), "
                      "evaluation.evaluate_pairs full-data C=1e6 fit",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path}")
    for delta, w in weights.items():
        acc = meta_predictors[delta]["accuracy"]
        print(f"  {delta}: weight={w:.4f} (accuracy {acc:.3f})")
    return payload


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Fit recommender feature weights on a corpus."
    )
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"CoNLL-U corpus path (default: {DEFAULT_CORPUS})")
    ap.add_argument("--scheme", default=DEFAULT_SCHEME,
                    choices=sorted(ROOT_POS_PRESETS),
                    help=f"annotation scheme (default: {DEFAULT_SCHEME})")
    ap.add_argument("--scorers", default=DEFAULT_SCORERS,
                    help=f"comma-separated scorer names (default: {DEFAULT_SCORERS})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output JSON path (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    scorer_names = [s.strip() for s in args.scorers.split(",") if s.strip()]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    corpus = args.corpus
    if not Path(corpus).is_absolute() and not Path(corpus).exists():
        corpus = str(_ROOT / corpus)

    fit_weights(corpus, args.scheme, scorer_names, out_path)


if __name__ == "__main__":
    main()
