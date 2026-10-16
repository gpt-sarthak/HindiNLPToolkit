"""
variants.word_order
===================
Word-order construction labels for a pairs table: which of the paper's
constructions each *reference* and each *variant* instantiates.

``generate_variants`` is surface-only — it knows every variant's constituent
order while building it but deliberately emits nothing beyond the surface
strings.  This module re-derives that order from the reference parse plus the
variant surface (``helpers.recover_permutation``, the same recovery every
positional scorer relies on) and labels both sides with
``helpers.classify_word_order``:

    Reference_Word_Order | Variant_Word_Order      each SOV / DOSV / IOSV

``OSV`` (object-fronted) is never stored — it is ``DOSV`` *or* ``IOSV``
(Ranjan & van Schijndel 2024, Table 1).  These columns are metadata, not
features: no scorer reads them and the central diff step ignores them.

    from variants import generate_variants, label_word_orders
    pairs_df = generate_variants(passed)
    pairs_df = pairs_df.join(label_word_orders(pairs_df, passed))
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from helpers import (
    classify_word_order,
    constituent_deprel,
    recover_permutation,
    reference_lookup,
)

WORD_ORDER_COLUMNS = ["Reference_Word_Order", "Variant_Word_Order"]


def label_word_orders(pairs_df: pd.DataFrame, passed: List[Dict]) -> pd.DataFrame:
    """
    Label every (reference, variant) row with its word-order constructions.

    Parameters
    ----------
    pairs_df : output of ``generate_variants`` (needs ``Ref_Index`` — or, for
               an older table, ``Sent_ID`` — and ``Variant_Sentence``; extra
               columns are ignored).
    passed   : the filter output the pairs were generated from — each item a
               dict with ``sentence`` / ``root_id`` / ``constituents``.

    Returns
    -------
    A DataFrame aligned to ``pairs_df.index`` with ``Reference_Word_Order`` and
    ``Variant_Word_Order`` (``'SOV'`` / ``'DOSV'`` / ``'IOSV'``), ready to
    ``join`` onto *pairs_df*.  Rows with no parse in *passed* get empty strings
    — never an exception.
    """
    if pairs_df.empty:
        return pd.DataFrame({c: [] for c in WORD_ORDER_COLUMNS}, index=pairs_df.index, dtype=object)

    # Per reference (keyed by the item's identity): constituents, deprels in
    # reference surface order, and the reference's own label — computed once.
    per_ref: Dict[int, Tuple[List[List[dict]], List[str], str]] = {}
    ref_col: List[str] = []
    var_col: List[str] = []

    for item, variant_sentence in zip(reference_lookup(pairs_df, passed),
                                      pairs_df["Variant_Sentence"]):
        if item is None:
            ref_col.append("")
            var_col.append("")
            continue
        if id(item) not in per_ref:
            consts = item["constituents"]
            deprels = [constituent_deprel(c, item["root_id"]) for c in consts]
            per_ref[id(item)] = (consts, deprels, classify_word_order(deprels))
        consts, deprels, ref_label = per_ref[id(item)]
        ref_col.append(ref_label)

        perm = recover_permutation(str(variant_sentence).split(" "), consts)
        var_col.append(classify_word_order([deprels[i] for i in perm]))

    return pd.DataFrame(
        {"Reference_Word_Order": ref_col, "Variant_Word_Order": var_col},
        index=pairs_df.index,
    )


__all__ = ["label_word_orders", "WORD_ORDER_COLUMNS"]
