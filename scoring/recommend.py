"""
scoring.recommend
=================
Pick the most natural word order among a sentence's candidates — the reference
plus every generated variant — from the raw true-role scores the selected
scorers already computed.

This is a *private-style* support module like ``scoring._lstm_common``: it
defines no :class:`~scoring.base.Scorer` subclass, so package auto-discovery
imports it harmlessly and registers nothing.  Use it through the package::

    from scoring import recommend_order

    result = recommend_order(scored_pairs_df, ["dependency_length", "lstm"])

How it ranks
------------
A model cannot be fit per sentence (a handful of unlabeled candidates), so the
combination is fixed in advance:

1. Each scorer's ``deltas()`` extractors read the raw reference/variant scores
   straight off the scored pairs table — nothing is recomputed.
2. Only deltas with a declared direction (``Scorer.higher_is_more_natural``) vote.
3. Per feature, candidate scores are z-scored *locally* (across the sentence's
   own candidates) and oriented so that higher always means "more natural".
4. A candidate's combined score is the weighted mean of its oriented z-scores;
   weights come from ``scoring/models/recommender_weights.json`` — standardized
   univariate logistic-regression coefficients fitted by
   ``python -m evaluation.fit_weights`` on a corpus (the Joachims pairwise
   trick: coefficients trained on (reference, variant) pairs rank single
   candidates directly).  All seven built-in scorers ship a fitted weight; any
   feature without a corpus-trained weight falls back to weight 1.0,
   direction-only.
5. The top-scoring candidate is recommended; exact ties go to the reference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "recommender_weights.json"

_weights: Optional[Dict[str, float]] = None  # lazy; {} when the file is absent


def _get_weights() -> Dict[str, float]:
    """Corpus-trained feature weights, ``{delta_name: weight}``.  Loaded once;
    a missing or unreadable file simply means every feature weighs 1.0."""
    global _weights
    if _weights is None:
        try:
            payload = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
            _weights = {k: float(v) for k, v in payload.get("weights", {}).items()}
        except (OSError, ValueError):
            _weights = {}
    return _weights


def _finite(value) -> Optional[float]:
    """*value* as a float, or None for anything non-numeric / NaN / ±inf."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _feature_specs(names: List[str]) -> List[dict]:
    """One spec per rankable delta of the named scorers: the delta name, the
    owning scorer, the direction, the weight, and the ``deltas()`` extractors.
    Deltas without a ``higher_is_more_natural`` entry are skipped (they don't vote)."""
    from . import get_scorers  # deferred: this module is imported by the package

    scorers = get_scorers()
    weights = _get_weights()
    specs: List[dict] = []
    for name in names:
        if name not in scorers:
            raise KeyError(
                f"Unknown scorer '{name}'. Available: {sorted(scorers)}"
            )
        scorer = scorers[name]
        # Read the new attribute; fall back to the pre-rename `higher_is_better`
        # so a teammate's un-migrated scorer keeps its recommendation vote.
        direction_map = scorer.higher_is_more_natural or getattr(
            scorer, "higher_is_better", {}
        )
        for delta, ref_fn, var_fn in scorer.deltas():
            direction = direction_map.get(delta)
            if direction is None:
                continue
            specs.append({
                "delta": delta,
                "scorer": scorer.name,
                "higher_is_more_natural": bool(direction),
                "weight": weights.get(delta, 1.0),
                "ref_fn": ref_fn,
                "var_fn": var_fn,
            })
    return specs


def _oriented_z(values: List[Optional[float]], higher_is_more_natural: bool) -> List[Optional[float]]:
    """Local z-scores of *values*, sign-flipped so higher = more natural.
    Entries stay None where the value is; everything is 0 when the feature
    cannot discriminate (fewer than 2 finite values, or zero spread)."""
    finite = [v for v in values if v is not None]
    if len(finite) < 2:
        return [None if v is None else 0.0 for v in values]
    mean = sum(finite) / len(finite)
    std = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))
    sign = 1.0 if higher_is_more_natural else -1.0
    return [
        None if v is None else (0.0 if std == 0 else sign * (v - mean) / std)
        for v in values
    ]


def recommend_order(pairs_df: pd.DataFrame, names: List[str]) -> dict:
    """
    Rank each sentence's candidate word orders and pick the most natural one.

    Parameters
    ----------
    pairs_df : the scored pairs table (``generate_variants`` output after
               ``apply_scorers``) — raw ``*_Reference`` / ``*_Variant`` columns
               must be present for the named scorers.
    names    : scorer names whose features should vote (the same list passed to
               ``apply_scorers``).  Raises KeyError for unknown names.

    Returns
    -------
    JSON-serializable dict::

        {"features":  [{delta, scorer, higher_is_more_natural, weight}, ...],
         "sentences": [{sent_id,
                        recommended: {id, sentence, is_reference, score} | None,
                        candidates: [{id, sentence, is_reference,
                                      values: {delta: float|None}, score}]}]}

    Candidates are listed reference-first (``id`` = ``"reference"``, then each
    ``Variant_ID``).  ``score`` is None for candidates with no usable feature
    value; ``recommended`` is None when no candidate could be scored.
    """
    specs = _feature_specs(names)
    features = [
        {k: s[k] for k in ("delta", "scorer", "higher_is_more_natural", "weight")}
        for s in specs
    ]
    result: dict = {"features": features, "sentences": []}
    if pairs_df.empty:
        return result

    # Group rows per source sentence, preserving first-appearance order.
    groups: Dict[str, List[dict]] = {}
    for row in pairs_df.to_dict("records"):
        groups.setdefault(row["Sent_ID"], []).append(row)

    for sent_id, rows in groups.items():
        # Reference first: its raw scores are identical on every row of the
        # group, so read them off the first one.
        candidates: List[dict] = [{
            "id": "reference",
            "sentence": str(rows[0]["Reference_Sentence"]),
            "is_reference": True,
            "values": {s["delta"]: _finite(s["ref_fn"](rows[0])) for s in specs},
        }]
        for row in rows:
            candidates.append({
                "id": str(row["Variant_ID"]),
                "sentence": str(row["Variant_Sentence"]),
                "is_reference": False,
                "values": {s["delta"]: _finite(s["var_fn"](row)) for s in specs},
            })

        # Weighted mean of oriented z-scores, over the features where each
        # candidate has a finite value.
        num = [0.0] * len(candidates)
        den = [0.0] * len(candidates)
        for s in specs:
            oriented = _oriented_z(
                [c["values"][s["delta"]] for c in candidates],
                s["higher_is_more_natural"],
            )
            for i, z in enumerate(oriented):
                if z is not None:
                    num[i] += s["weight"] * z
                    den[i] += s["weight"]

        best = None
        for cand, n, d in zip(candidates, num, den):
            cand["score"] = (n / d) if d > 0 else None
            # Strict ">" + reference-first order: exact ties go to the reference.
            if cand["score"] is not None and (best is None or cand["score"] > best["score"]):
                best = cand

        result["sentences"].append({
            "sent_id": str(sent_id),
            "recommended": None if best is None else {
                k: best[k] for k in ("id", "sentence", "is_reference", "score")
            },
            "candidates": candidates,
        })
    return result
