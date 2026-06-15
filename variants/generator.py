"""
variants.generator
==================
Generates grammatically valid permutations of preverbal constituents for each
filtered sentence and pairs them with the corpus (reference) order.

This module is intentionally *surface only*: it produces the (reference,
variant) sentence pairs and the ML pairing label, but computes **no** features.
Feature extraction (dependency length, information status, …) lives in the
``scoring`` package, where each scorer adds its own columns and the per-feature
diffs are computed from ``ML_Label`` at the end of the pipeline.

Public functions
----------------
generate_variants(passed_list, valid_deprel_pairs=None, max_variants=99,
                  seed=42, output_dir=None) -> pandas.DataFrame
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


_COLUMNS = [
    "Sent_ID",
    "Variant_ID",
    "ML_Label",
    "Reference_Sentence",
    "Variant_Sentence",
]


# ---------------------------------------------------------------------------
# Grammar-filter helpers
# ---------------------------------------------------------------------------

def _get_constituent_deprel(constituent: List[Dict], root_id: int) -> str:
    """
    Return the deprel of the token inside *constituent* that directly attaches
    to *root_id*.  Falls back to 'UNKNOWN' if no such token is found.
    """
    for tok in constituent:
        if tok["head"] == root_id:
            return tok["deprel"]
    return "UNKNOWN"


def _is_valid_perm(
    perm: Tuple[int, ...],
    orig_deprels: List[str],
    valid_deprel_pairs: Set[Tuple[str, str]],
) -> bool:
    """
    Return True if every adjacent (deprel_A, deprel_B) pair produced by this
    permutation exists in valid_deprel_pairs.

    This grammar filter ensures generated variants respect the bigram
    co-occurrence patterns observed in the corpus, keeping them linguistically
    plausible even when constituent order is shuffled.
    """
    permuted = [orig_deprels[i] for i in perm]
    return all(
        (permuted[i], permuted[i + 1]) in valid_deprel_pairs
        for i in range(len(permuted) - 1)
    )


# ---------------------------------------------------------------------------
# Public: generate_variants
# ---------------------------------------------------------------------------

def generate_variants(
    passed_list: List[Dict],
    valid_deprel_pairs: Optional[Set[Tuple[str, str]]] = None,
    max_variants: int = 99,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate grammatically valid preverbal constituent permutations for each
    sentence and pair them with the corpus (reference) order.

    How it works
    ------------
    For each passed sentence:
    1.  The preverbal constituents are taken from *passed_list* (computed by
        ``filter_sentences``).
    2.  Every permutation of those constituents is checked against
        ``valid_deprel_pairs``.  A permutation passes if every adjacent pair of
        dependency-relation labels it produces has been observed in the corpus,
        making it linguistically plausible.
    3.  Up to ``max_variants`` passing permutations are kept as variants.
    4.  Each (reference, variant) pair is emitted as one row, with an
        alternating ``ML_Label`` (the *flip signal*): on label==1 rows the pair
        is presented reference-first, on label==0 rows it is flipped
        (variant-first).  This produces an exactly balanced 50/50 split and is
        what every feature scorer later uses to orient its diff.

    No features are computed here — see the ``scoring`` package.

    Grammar filter (valid_deprel_pairs)
    ------------------------------------
    When ``valid_deprel_pairs=None`` (default), the set is built from all
    adjacent deprel pairs observed across *passed_list*.  For a single sentence
    this is intentionally restrictive.  Pass a pre-computed set from a larger
    corpus to relax the filter.

    Permutation strategy
    --------------------
    - N! <= 100,000 : exhaustive enumeration of all permutations.
    - N! >  100,000 : random sampling (up to 500 valid permutations,
                      100,000 attempts) to avoid combinatorial explosion.

    Parameters
    ----------
    passed_list : List[Dict]
        Output of ``filter_sentences()[0]``.
    valid_deprel_pairs : Set[Tuple[str, str]], optional
        Adjacent (deprel_A, deprel_B) bigrams considered grammatically licit.
        Built from *passed_list* if not given.
    max_variants : int
        Maximum variant permutations retained per sentence (default 99).
    seed : int
        Random seed for reproducible sampling (default 42).
    output_dir : str, optional
        If given, writes ``reference_variant_pairs.csv`` to this directory.

    Returns
    -------
    pairs_df : pandas.DataFrame
        Columns: Sent_ID | Variant_ID | ML_Label | Reference_Sentence |
        Variant_Sentence.  ``Variant_ID`` is ``{Sent_ID}_v{n}``.

    Notes
    -----
    A tiny fraction of pairs may have identical surface text when a sentence
    contains reduplicated words as separate constituents (e.g. "हंसते हंसते").
    This is expected and harmless.

    Example
    -------
    >>> from variants import generate_variants
    >>> pairs_df = generate_variants(passed, output_dir="out/")
    """
    if not passed_list:
        return pd.DataFrame(columns=_COLUMNS)

    rng = random.Random(seed)

    # Build valid_deprel_pairs from corpus adjacencies if not supplied.
    if valid_deprel_pairs is None:
        valid_deprel_pairs = set()
        for item in passed_list:
            deprels = [_get_constituent_deprel(c, item["root_id"]) for c in item["constituents"]]
            for i in range(len(deprels) - 1):
                valid_deprel_pairs.add((deprels[i], deprels[i + 1]))

    pairs_rows: List[Dict] = []
    # Alternating toggle keeps the label distribution exactly 50/50 across the
    # full dataset rather than relying on random chance.
    toggle_label = True

    for item in passed_list:
        original_sent = item["sentence"]
        real_consts = item["constituents"]
        root_id = item["root_id"]

        sent_id = original_sent.metadata.get("sent_id", "Unknown_ID")
        ref_text = " ".join(tok["form"] for tok in original_sent)

        # Split the sentence into three non-overlapping regions:
        #   prefix — tokens before the preverbal block (usually empty)
        #   middle — the preverbal constituents (reordered per permutation)
        #   suffix — root + all postverbal tokens (held fixed)
        flat_preverbal_ids = {tok["id"] for const in real_consts for tok in const}
        if not flat_preverbal_ids:
            continue
        start_pos = min(flat_preverbal_ids)
        prefix = [t for t in original_sent if t["id"] < start_pos and t["id"] not in flat_preverbal_ids]
        suffix = [t for t in original_sent if t["id"] > start_pos and t["id"] not in flat_preverbal_ids]

        indices = list(range(len(real_consts)))
        orig_deprels = [_get_constituent_deprel(c, root_id) for c in real_consts]
        total_possible = math.factorial(len(indices))

        # Enumerate or sample permutations.  The 100,000 threshold avoids
        # memory/time issues for sentences with 8+ preverbal constituents.
        if total_possible <= 100_000:
            grammatical_perms = [
                perm
                for perm in itertools.permutations(indices)
                if list(perm) != indices
                and _is_valid_perm(perm, orig_deprels, valid_deprel_pairs)
            ]
        else:
            seen: Set[Tuple[int, ...]] = set()
            attempts = 0
            while len(seen) < 500 and attempts < 100_000:
                attempts += 1
                perm = tuple(rng.sample(indices, len(indices)))
                if list(perm) == indices or perm in seen:
                    continue
                if _is_valid_perm(perm, orig_deprels, valid_deprel_pairs):
                    seen.add(perm)
            grammatical_perms = list(seen)

        if len(grammatical_perms) > max_variants:
            selected_perms = rng.sample(grammatical_perms, max_variants)
        else:
            selected_perms = grammatical_perms

        for var_idx, perm in enumerate(selected_perms, start=1):
            permuted_consts = [real_consts[i] for i in perm]
            new_order = prefix + [tok for const in permuted_consts for tok in const] + suffix
            var_text = " ".join(tok["form"] for tok in new_order)

            label = 1 if toggle_label else 0
            toggle_label = not toggle_label

            pairs_rows.append({
                "Sent_ID": sent_id,
                "Variant_ID": f"{sent_id}_v{var_idx}",
                "ML_Label": label,
                "Reference_Sentence": ref_text,
                "Variant_Sentence": var_text,
            })

    pairs_df = pd.DataFrame(pairs_rows, columns=_COLUMNS)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pairs_df.to_csv(out / "reference_variant_pairs.csv", index=False, encoding="utf-8")

    return pairs_df
