"""
filtering.filters
=================
Individual filter functions for each of the seven linguistic quality checks,
plus a combined pipeline (filter_sentences) and a summary function.

Individual filters
------------------
filter_questions(sentences, check_punct, check_pos, check_features)
filter_negatives(sentences, check_pos, check_features)
filter_ghost_ids(sentences)
filter_non_projective(sentences)
filter_bad_root(sentences, allowed_pos)
filter_punct_constituents(sentences_or_items, allowed_pos)
filter_min_phrases(sentences_or_items, min_phrases, allowed_pos,
                   subject_deprels, object_deprels, require_core_args)

Combined pipeline
-----------------
filter_sentences(sentences, allowed_root_pos, min_phrases, output_dir,
                 subject_deprels, object_deprels, require_core_args)
    Runs all seven filters in order.  Returns (passed_list, rejected_df, passed_df).

Summary
-------
summarize(passed_list, rejected_df)
    Returns a dict of per-filter rejection counts and corpus statistics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from conllu import TokenList

# Default POS tags accepted on the sentence root for preverbal analysis.
_DEFAULT_ALLOWED_POS: Tuple[str, ...] = ("VERB", "AUX", "VM", "VAUX")

# Core-argument relations, unioned across both annotation schemes (UD first,
# Paninian second).  A union is safe because the two schemes share no deprel
# label and are never mixed in one call, so no scheme flag is needed here —
# the same reasoning behind _DEFAULT_ALLOWED_POS above.
# Deliberately excluded: `k1s` (predicative complement of a copula, not a
# subject) and `pof` (the nominal half of a conjunct verb, not an object).
_DEFAULT_SUBJECT_DEPRELS: Tuple[str, ...] = ("nsubj", "nsubj:pass", "k1")
_DEFAULT_OBJECT_DEPRELS: Tuple[str, ...] = ("obj", "iobj", "k2", "k4")

_REJECTED_COLS = ["Sent_ID", "Sentence", "Reason"]


# ---------------------------------------------------------------------------
# Internal tree helpers
# ---------------------------------------------------------------------------

def _sent_meta(sent: TokenList) -> Tuple[str, str]:
    """Return (sent_id, text) from a TokenList's metadata."""
    sent_id = sent.metadata.get("sent_id", "Unknown_ID")
    text = sent.metadata.get(
        "text", " ".join(tok["form"] for tok in sent if isinstance(tok["id"], int))
    )
    return sent_id, text


def _make_rejected_df(rows: List[Dict]) -> pd.DataFrame:
    """Build a uniform rejected-sentences DataFrame from a list of row dicts."""
    return (
        pd.DataFrame(rows, columns=_REJECTED_COLS)
        if rows
        else pd.DataFrame(columns=_REJECTED_COLS)
    )


def _get_root_token(sentence: List[Dict]) -> Optional[Dict]:
    """Return the token whose head is 0 (the syntactic root), or None."""
    for tok in sentence:
        if tok["head"] == 0:
            return tok
    return None


def _get_subtree_ids(sentence: List[Dict], head_id: int) -> List[int]:
    """
    Recursively collect the IDs of all tokens in the subtree rooted at
    head_id.  Returns a sorted, deduplicated list including head_id itself.
    """
    ids = [head_id]
    for tok in sentence:
        if tok["head"] == head_id:
            ids.extend(_get_subtree_ids(sentence, tok["id"]))
    return sorted(set(ids))


def _get_constituent_deprel(constituent: List[Dict], root_id: int) -> str:
    """
    Return the deprel of the token inside *constituent* that directly attaches
    to *root_id*.  Falls back to 'UNKNOWN' if no such token is found.
    """
    for tok in constituent:
        if tok["head"] == root_id:
            return tok["deprel"]
    return "UNKNOWN"


def _is_projective(sentence: List[Dict]) -> bool:
    """
    Return True if the dependency tree has no crossing arcs (projective).
    Two arcs (u1,v1) and (u2,v2) cross when one endpoint of one arc lies
    strictly between the endpoints of the other.
    """
    edges = []
    for tok in sentence:
        if isinstance(tok["id"], int) and isinstance(tok["head"], int) and tok["head"] != 0:
            u, v = tok["id"], tok["head"]
            edges.append((min(u, v), max(u, v)))
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            if (u1 < u2 < v1 < v2) or (u2 < u1 < v2 < v1):
                return False
    return True


def _get_preverbal_constituents(
    sentence: List[Dict],
    allowed_pos: Tuple[str, ...] = _DEFAULT_ALLOWED_POS,
) -> Tuple[List[List[Dict]], Optional[int], Optional[str]]:
    """
    Extract preverbal constituents from a sentence.

    A preverbal constituent is the full dependency subtree of any direct
    dependent of the root that appears to the left of the root (token_id < root_id).

    Parameters
    ----------
    sentence    : list of token dicts
    allowed_pos : POS tags accepted on the root token

    Returns (constituents, root_id, error_message).
    error_message is None on success, or a rejection-reason string on failure.
    """
    root = _get_root_token(sentence)
    if root is None:
        return [], None, "Non-verbal root: No root token found"
    if root["upos"] not in allowed_pos:
        allowed_str = "/".join(allowed_pos)
        return [], None, f"Non-verbal root: Root POS is '{root['upos']}', expected {allowed_str}"
    root_id = root["id"]
    preverbal = []
    for dep in [tok for tok in sentence if tok["head"] == root_id and tok["id"] < root_id]:
        subtree_ids = set(_get_subtree_ids(sentence, dep["id"]))
        preverbal.append([tok for tok in sentence if tok["id"] in subtree_ids])
    return preverbal, root_id, None


# ---------------------------------------------------------------------------
# Public: individual filter functions
# ---------------------------------------------------------------------------

def filter_questions(
    sentences: List[TokenList],
    check_punct: bool = True,
    check_pos: bool = True,
    check_features: bool = True,
) -> Tuple[List[TokenList], pd.DataFrame]:
    """
    Remove interrogative sentences.

    Three independent signals are checked; a sentence is rejected if any
    enabled signal fires.  All three are on by default because annotation
    conventions vary across Hindi corpora.

    Parameters
    ----------
    sentences      : list of TokenList objects
    check_punct    : reject if ``?`` appears anywhere in the surface text.
    check_pos      : reject if any token carries the ``WQ`` POS tag
                     (interrogative word, Paninian tagset).
    check_features : reject if any token has ``PronType=Int`` morphological
                     feature (interrogative pronoun, UD convention).

    Returns
    -------
    passed     : List[TokenList] — sentences that are not questions.
    rejected   : DataFrame with columns Sent_ID | Sentence | Reason.

    Example
    -------
    >>> passed, rej = filter_questions(sentences)
    >>> passed, rej = filter_questions(sentences, check_punct=False)  # skip ? check
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    passed: List[TokenList] = []
    rejected: List[Dict] = []

    for sent in sentences:
        sent_id, text = _sent_meta(sent)
        is_question = check_punct and "?" in text

        for tok in sent:
            if check_pos and tok.get("upos") == "WQ":
                is_question = True
            if check_features:
                feats = tok.get("feats") or {}
                if isinstance(feats, dict) and feats.get("PronType") == "Int":
                    is_question = True

        if is_question:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": "Interrogative"})
        else:
            passed.append(sent)

    return passed, _make_rejected_df(rejected)


def filter_negatives(
    sentences: List[TokenList],
    check_pos: bool = True,
    check_features: bool = True,
) -> Tuple[List[TokenList], pd.DataFrame]:
    """
    Remove sentences that contain negation.

    Negation interacts with constituent order in ways outside the scope of
    the preverbal-ordering model, so negative sentences are excluded.

    Parameters
    ----------
    sentences      : list of TokenList objects
    check_pos      : reject if any token carries the ``NEG`` POS tag.
    check_features : reject if any token has ``Polarity=Neg`` morphological
                     feature (UD convention).

    Returns
    -------
    passed   : List[TokenList]
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.

    Example
    -------
    >>> passed, rej = filter_negatives(sentences)
    >>> passed, rej = filter_negatives(sentences, check_features=False)
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    passed: List[TokenList] = []
    rejected: List[Dict] = []

    for sent in sentences:
        sent_id, text = _sent_meta(sent)
        is_negative = False

        for tok in sent:
            if check_pos and tok.get("upos") == "NEG":
                is_negative = True
            if check_features:
                feats = tok.get("feats") or {}
                if isinstance(feats, dict) and feats.get("Polarity") == "Neg":
                    is_negative = True

        if is_negative:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": "Negation"})
        else:
            passed.append(sent)

    return passed, _make_rejected_df(rejected)


def filter_ghost_ids(
    sentences: List[TokenList],
) -> Tuple[List[TokenList], pd.DataFrame]:
    """
    Remove sentences that contain non-integer (ghost/empty) token IDs.

    CoNLL-U allows fractional IDs such as ``1.1`` for empty nodes in enhanced
    dependency graphs.  These break all integer-position arithmetic used
    downstream, so any sentence containing them is rejected.

    Parameters
    ----------
    sentences : list of TokenList objects

    Returns
    -------
    passed   : List[TokenList]
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.

    Example
    -------
    >>> passed, rej = filter_ghost_ids(sentences)
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    passed: List[TokenList] = []
    rejected: List[Dict] = []

    for sent in sentences:
        sent_id, text = _sent_meta(sent)
        if any(not isinstance(tok["id"], int) for tok in sent):
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": "Empty nodes"})
        else:
            passed.append(sent)

    return passed, _make_rejected_df(rejected)


def filter_non_projective(
    sentences: List[TokenList],
) -> Tuple[List[TokenList], pd.DataFrame]:
    """
    Remove sentences whose dependency tree contains crossing arcs.

    A projective tree has no crossing arcs.  Non-projective structures
    indicate long-distance dependencies that the constituent-block
    permutation model cannot represent faithfully.

    Parameters
    ----------
    sentences : list of TokenList objects

    Returns
    -------
    passed   : List[TokenList]
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.

    Example
    -------
    >>> passed, rej = filter_non_projective(sentences)
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    passed: List[TokenList] = []
    rejected: List[Dict] = []

    for sent in sentences:
        sent_id, text = _sent_meta(sent)
        if not _is_projective(sent):
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": "Non-projective dependencies"})
        else:
            passed.append(sent)

    return passed, _make_rejected_df(rejected)


def filter_bad_root(
    sentences: List[TokenList],
    allowed_pos: Optional[List[str]] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Remove sentences whose dependency root is missing or carries a non-verbal
    POS tag, then extract preverbal constituents for sentences that pass.

    Preverbal constituent extraction is only well-defined for verb-headed
    clauses.  Sentences with a nominal, adverbial, or missing root are
    excluded.

    Parameters
    ----------
    sentences   : list of TokenList objects
    allowed_pos : POS tags accepted on the root token.
                  Defaults to ``["VERB", "AUX", "VM", "VAUX"]``.
                  Extend this list if your corpus uses additional verbal tags.

    Returns
    -------
    passed   : List[Dict]  — each dict has keys:
               ``sentence`` (TokenList), ``root_id`` (int),
               ``constituents`` (List[List[Dict]]).
               This is a valid input for ``filter_min_phrases()`` and
               ``generate_variants()``.
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.
               The Reason field includes the actual root POS tag found
               (e.g. ``"Non-verbal root: Root POS is 'NN', expected VERB/AUX/VM/VAUX"``).

    Example
    -------
    >>> passed, rej = filter_bad_root(sentences)
    >>> passed, rej = filter_bad_root(sentences, allowed_pos=["VERB", "AUX", "VM", "VAUX", "VNN"])
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    pos_tuple = tuple(allowed_pos) if allowed_pos else _DEFAULT_ALLOWED_POS

    passed: List[Dict] = []
    rejected: List[Dict] = []

    for sent in sentences:
        sent_id, text = _sent_meta(sent)
        consts, root_id, error = _get_preverbal_constituents(sent, allowed_pos=pos_tuple)
        if error:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": error})
        else:
            passed.append({"sentence": sent, "root_id": root_id, "constituents": consts})

    return passed, _make_rejected_df(rejected)


def filter_punct_constituents(
    sentences_or_items: Union[List[TokenList], List[Dict]],
    allowed_pos: Optional[List[str]] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Remove sentences that contain a bare punctuation token as a preverbal
    constituent.

    A preverbal constituent is considered "punct-attached" when the token
    inside that constituent that directly depends on the root carries
    ``deprel == "punct"``.  In practice this is always a single comma,
    semicolon, or quotation mark that the annotator attached directly to the
    verb — typically the comma in discourse-connector + comma openers such as
    ``"लेकिन,"`` or ``"बहरहाल,"``.

    Why this matters
    ----------------
    Constituent permutation treats every constituent as a free-floating block.
    When one of those blocks is a lone comma, every non-reference permutation
    produces ungrammatical output — the comma migrates to the middle or start
    of the sentence:

        Reference : लेकिन , उसकी दृढ़ इच्छा के आगे उन्हें झुकना पड़ा ।
        Variant   : उसकी दृढ़ इच्छा के आगे , लेकिन उन्हें झुकना पड़ा ।  ✗

    All 62 affected sentences in UD Hindi-HDTB were found to produce zero
    clean variants, so rejecting them is the correct response.

    Parameters
    ----------
    sentences_or_items : List[TokenList] or List[Dict]
        Raw CoNLL-U sentences or enriched dicts from ``filter_bad_root``.
        When passing raw TokenList objects, constituent extraction is performed
        internally (same logic as ``filter_bad_root``).
    allowed_pos        : only used when input is List[TokenList].  POS tags
                         accepted on the root (default: VERB/AUX/VM/VAUX).

    Returns
    -------
    passed   : List[Dict]  — each dict has keys ``sentence``, ``root_id``,
               ``constituents``.  Ready for ``filter_min_phrases()`` and
               ``generate_variants()``.
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.
               Reason is always ``"Punctuation-attached constituent"``.

    Raises
    ------
    ValueError : if *sentences_or_items* is empty.

    Example
    -------
    >>> # Independent use
    >>> passed, rej = filter_punct_constituents(sentences)

    >>> # Chained after filter_bad_root
    >>> items, _ = filter_bad_root(sentences)
    >>> passed, rej = filter_punct_constituents(items)
    """
    if not sentences_or_items:
        raise ValueError("sentences_or_items list is empty.")

    pos_tuple = tuple(allowed_pos) if allowed_pos else _DEFAULT_ALLOWED_POS
    pre_computed = isinstance(sentences_or_items[0], dict)

    passed: List[Dict] = []
    rejected: List[Dict] = []

    for item in sentences_or_items:
        if pre_computed:
            sent = item["sentence"]
            root_id = item["root_id"]
            consts = item["constituents"]
            sent_id, text = _sent_meta(sent)
            error = None
        else:
            sent = item
            sent_id, text = _sent_meta(sent)
            consts, root_id, error = _get_preverbal_constituents(sent, allowed_pos=pos_tuple)

        if error:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": error})
            continue

        has_punct_constituent = any(
            tok["deprel"] == "punct"
            for const in consts
            for tok in const
            if tok["head"] == root_id
        )

        if has_punct_constituent:
            rejected.append({
                "Sent_ID": sent_id,
                "Sentence": text,
                "Reason": "Punctuation-attached constituent",
            })
        else:
            passed.append({"sentence": sent, "root_id": root_id, "constituents": consts})

    return passed, _make_rejected_df(rejected)


def filter_min_phrases(
    sentences_or_items: Union[List[TokenList], List[Dict]],
    min_phrases: int = 2,
    allowed_pos: Optional[List[str]] = None,
    subject_deprels: Optional[List[str]] = None,
    object_deprels: Optional[List[str]] = None,
    require_core_args: bool = True,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Remove sentences that lack a usable preverbal domain: fewer than
    *min_phrases* preverbal constituents, or (by default) missing a preverbal
    subject or object.

    Two conditions, because the preverbal block has to support the actual
    research question:

    1. **Count** — variant generation requires at least two preverbal
       constituents so that at least one non-trivial permutation exists.
    2. **Core arguments** — the constituents must include both a subject and an
       object.  This is criterion (a) of Ranjan & van Schijndel (2024),
       *"the trees contain both well-defined subjects and objects"*, and it is
       what makes the SOV/OSV canonicity contrast meaningful: a sentence of a
       subject plus three obliques satisfies the count but has no object to
       move, so permuting it answers no question about word order.

    Both are checked here rather than in a separate filter because the count was
    only ever a proxy for this.  Set ``require_core_args=False`` to get the old
    count-only behaviour.

    This function accepts two input types so it can be used independently or
    chained after ``filter_bad_root``:

    * **List[TokenList]** — constituents are extracted internally using the same
      root-detection logic as ``filter_bad_root``.  Sentences with a bad root
      are rejected with the root error as the reason.
    * **List[Dict]** — pre-computed output of ``filter_bad_root`` (already has
      ``root_id`` and ``constituents``).  No re-extraction is performed.

    Parameters
    ----------
    sentences_or_items : List[TokenList] or List[Dict]
        Raw CoNLL-U sentences or enriched dicts from ``filter_bad_root``.
    min_phrases        : minimum number of preverbal constituents required
                         (default 2, must be ≥ 1).
    allowed_pos        : only used when input is List[TokenList].  POS tags
                         accepted on the root (default: VERB/AUX/VM/VAUX).
    subject_deprels    : relations counted as a subject.  Defaults to
                         ``("nsubj", "nsubj:pass", "k1")`` — both schemes at
                         once, which is safe because they share no label.
    object_deprels     : relations counted as an object.  Defaults to
                         ``("obj", "iobj", "k2", "k4")``; indirect objects
                         count.  Pass ``["obj", "k2"]`` for direct objects only.
    require_core_args  : when True (default) also require a preverbal subject
                         and object.  False restores the old count-only
                         behaviour.

    Returns
    -------
    passed   : List[Dict]  — each dict has keys ``sentence``, ``root_id``,
               ``constituents``.  Ready for ``generate_variants()``.
    rejected : DataFrame with columns Sent_ID | Sentence | Reason.
               Reason is ``"Fewer than N preverbal constituent(s)"`` for a count
               failure, or one of ``"Missing preverbal subject"`` /
               ``"Missing preverbal object"`` /
               ``"Missing preverbal subject and object"``.

    Raises
    ------
    ValueError : if *sentences_or_items* is empty, or *min_phrases* < 1.

    Example
    -------
    >>> # Independent use on raw sentences
    >>> passed, rej = filter_min_phrases(sentences, min_phrases=3)

    >>> # Chained after filter_bad_root (no re-extraction)
    >>> items, _ = filter_bad_root(sentences)
    >>> passed, rej = filter_min_phrases(items, min_phrases=2)
    """
    if not sentences_or_items:
        raise ValueError("sentences_or_items list is empty.")
    if min_phrases < 1:
        raise ValueError(f"min_phrases must be >= 1, got {min_phrases}.")

    pos_tuple = tuple(allowed_pos) if allowed_pos else _DEFAULT_ALLOWED_POS
    subj_set = set(subject_deprels) if subject_deprels else set(_DEFAULT_SUBJECT_DEPRELS)
    obj_set = set(object_deprels) if object_deprels else set(_DEFAULT_OBJECT_DEPRELS)
    # Detect input type from the first element
    pre_computed = isinstance(sentences_or_items[0], dict)

    passed: List[Dict] = []
    rejected: List[Dict] = []
    reason_template = f"Fewer than {min_phrases} preverbal constituent{'s' if min_phrases != 1 else ''}"

    for item in sentences_or_items:
        if pre_computed:
            sent = item["sentence"]
            root_id = item["root_id"]
            consts = item["constituents"]
            sent_id, text = _sent_meta(sent)
            error = None
        else:
            sent = item
            sent_id, text = _sent_meta(sent)
            consts, root_id, error = _get_preverbal_constituents(sent, allowed_pos=pos_tuple)

        if error:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": error})
            continue
        if len(consts) < min_phrases:
            rejected.append({"Sent_ID": sent_id, "Sentence": text, "Reason": reason_template})
            continue

        if require_core_args:
            # Every constituent is the subtree of one direct root dependent, so
            # its deprel *is* that dependent's relation to the root.
            deprels = {_get_constituent_deprel(c, root_id) for c in consts}
            missing = []
            if not (deprels & subj_set):
                missing.append("subject")
            if not (deprels & obj_set):
                missing.append("object")
            if missing:
                rejected.append({
                    "Sent_ID": sent_id,
                    "Sentence": text,
                    "Reason": f"Missing preverbal {' and '.join(missing)}",
                })
                continue

        passed.append({"sentence": sent, "root_id": root_id, "constituents": consts})

    return passed, _make_rejected_df(rejected)


# ---------------------------------------------------------------------------
# Public: combined pipeline
# ---------------------------------------------------------------------------

def filter_sentences(
    sentences: List[TokenList],
    allowed_root_pos: Optional[List[str]] = None,
    min_phrases: int = 2,
    output_dir: Optional[str] = None,
    subject_deprels: Optional[List[str]] = None,
    object_deprels: Optional[List[str]] = None,
    require_core_args: bool = True,
) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    Run all seven filters in sequence and return the final passed set plus
    a merged rejection log.

    This is the recommended entry point for full pipeline use.  Each filter
    can also be called individually (see module-level docstring).

    Pipeline order
    --------------
    1. ``filter_questions``           — removes interrogative sentences
    2. ``filter_negatives``           — removes sentences with negation
    3. ``filter_ghost_ids``           — removes sentences with non-integer token IDs
    4. ``filter_non_projective``      — removes non-projective dependency trees
    5. ``filter_bad_root``            — removes non-verbal roots; extracts constituents
    6. ``filter_punct_constituents``  — removes sentences with a bare punct constituent
    7. ``filter_min_phrases``         — removes sentences below the constituent-count
                                        threshold, or missing a preverbal subject/object

    Parameters
    ----------
    sentences        : list of TokenList objects (output of ``load_input``).
    allowed_root_pos : POS tags accepted on the sentence root for filter 5.
                       Defaults to ``["VERB", "AUX", "VM", "VAUX"]``.
    min_phrases      : minimum preverbal constituents required for filter 7
                       (default 2).
    output_dir       : if given, writes ``passed_sentences.csv`` and
                       ``rejected_sentences.csv`` to this directory.
    subject_deprels  : filter 7 — relations counted as a subject
                       (default ``("nsubj", "nsubj:pass", "k1")``).
    object_deprels   : filter 7 — relations counted as an object
                       (default ``("obj", "iobj", "k2", "k4")``).
    require_core_args: filter 7 — when True (default) also require a preverbal
                       subject and object.  False gives count-only filtering.

    Returns
    -------
    passed_list  : List[Dict]  — dicts with ``sentence``, ``root_id``,
                   ``constituents``.  Direct input to ``generate_variants()``.
    rejected_df  : DataFrame   — all rejected sentences across all seven filters,
                   columns: Sent_ID | Sentence | Reason.
    passed_df    : DataFrame   — rich feature table for the passed sentences,
                   columns: Sent_ID | Sentence | Root_ID | Constituent_Count |
                   Character_Length | Sentence_Length | Constituent_Lengths |
                   Deprel_Tags | Grammatical_Pairs.

    Raises
    ------
    ValueError : if *sentences* is empty, or *min_phrases* < 1.

    Example
    -------
    >>> from filtering import filter_sentences
    >>> passed, rejected_df, passed_df = filter_sentences(sentences, output_dir="out/")
    >>> # Custom root POS and minimum phrase count
    >>> passed, rej, _ = filter_sentences(sentences, allowed_root_pos=["VERB", "AUX"], min_phrases=3)
    """
    if not sentences:
        raise ValueError("sentences list is empty.")

    # ── Run all seven filters in sequence ────────────────────────────────────
    # Each individual filter raises on an empty input list (a deliberate guard
    # for standalone use).  In the pipeline a stage can legitimately empty the
    # set — e.g. the only sentence has a bad root — so skip any later stage whose
    # input is already empty and still emit the rejection log + summary below.
    # (Without this, one rejected sentence aborts the whole run with no outputs.)
    _empty = _make_rejected_df([])
    p1, r1 = filter_questions(sentences)
    p2, r2 = filter_negatives(p1) if p1 else ([], _empty)
    p3, r3 = filter_ghost_ids(p2) if p2 else ([], _empty)
    p4, r4 = filter_non_projective(p3) if p3 else ([], _empty)
    p5, r5 = filter_bad_root(p4, allowed_pos=allowed_root_pos) if p4 else ([], _empty)
    # p5 onward is List[Dict] with pre-computed constituents
    p6, r6 = filter_punct_constituents(p5) if p5 else ([], _empty)
    passed_list, r7 = filter_min_phrases(
        p6,
        min_phrases=min_phrases,
        subject_deprels=subject_deprels,
        object_deprels=object_deprels,
        require_core_args=require_core_args,
    ) if p6 else ([], _empty)

    # ── Merge all rejection logs into one DataFrame ───────────────────────────
    rejected_df = pd.concat([r1, r2, r3, r4, r5, r6, r7], ignore_index=True)

    # ── Build passed_df — rich feature table for downstream analysis ──────────
    passed_rows: List[Dict] = []
    for item in passed_list:
        sent_obj = item["sentence"]
        root_id = item["root_id"]
        consts = item["constituents"]
        sent_id, sent_text = _sent_meta(sent_obj)
        deprels = [_get_constituent_deprel(c, root_id) for c in consts]
        # Adjacent deprel pairs capture the grammatical sequencing of each ordering.
        grammatical_pairs = [(deprels[i], deprels[i + 1]) for i in range(len(deprels) - 1)]
        passed_rows.append({
            "Sent_ID": sent_id,
            "Sentence": sent_text,
            "Root_ID": root_id,
            "Constituent_Count": len(consts),
            "Character_Length": len(sent_text),
            "Sentence_Length": sum(1 for tok in sent_obj if isinstance(tok["id"], int)),
            "Constituent_Lengths": str([len(c) for c in consts]),
            "Deprel_Tags": str(deprels),
            "Grammatical_Pairs": str(grammatical_pairs),
        })

    _cols = [
        "Sent_ID", "Sentence", "Root_ID", "Constituent_Count",
        "Character_Length", "Sentence_Length",
        "Constituent_Lengths", "Deprel_Tags", "Grammatical_Pairs",
    ]
    passed_df = (
        pd.DataFrame(passed_rows, columns=_cols)
        if passed_rows
        else pd.DataFrame(columns=_cols)
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        passed_df.to_csv(out / "passed_sentences.csv", index=False, encoding="utf-8")
        rejected_df.to_csv(out / "rejected_sentences.csv", index=False, encoding="utf-8")

    return passed_list, rejected_df, passed_df


# ---------------------------------------------------------------------------
# Public: summarize
# ---------------------------------------------------------------------------

def summarize(
    passed_list: List[Dict],
    rejected_df: pd.DataFrame,
) -> Dict:
    """
    Produce a summary of filter results and basic corpus statistics.

    Works with the output of ``filter_sentences`` or any combination of
    individual filter outputs whose rejected DataFrames have been concatenated.

    Parameters
    ----------
    passed_list : List[Dict]   — output of the final filter stage.
    rejected_df : pd.DataFrame — merged rejection log.

    Returns
    -------
    dict with the following keys
    ----------------------------
    total_input               : int   — sentences fed into the first filter
    total_passed              : int   — sentences that cleared all filters
    total_rejected            : int   — sentences rejected by any filter
    pass_rate                 : float — total_passed / total_input
    rejection_counts          : dict  — {reason_string: count}
    filter_order              : list  — [{filter, rejected}, …] in pipeline order
    avg_sentence_length       : float — mean token count of passed sentences
    avg_constituent_count          : float — mean preverbal constituent count
    constituent_count_distribution : dict  — {n_constituents: sentence_count}

    Example
    -------
    >>> from filtering import summarize
    >>> print(summarize(passed, rejected_df))
    """
    total_rejected = len(rejected_df)
    total_passed = len(passed_list)
    total_input = total_passed + total_rejected

    rejection_counts: Dict[str, int] = {}
    if not rejected_df.empty and "Reason" in rejected_df.columns:
        rejection_counts = rejected_df["Reason"].value_counts().to_dict()

    # Group all "Non-verbal root: …" variants under one entry so filter_order has
    # exactly one entry per filter stage.
    filter_order = []
    for key in ("Interrogative", "Negation", "Empty nodes",
                "Non-projective dependencies", "Non-verbal root",
                "Punctuation-attached constituent", "Fewer than"):
        if key == "Non-verbal root":
            count = sum(v for k, v in rejection_counts.items() if k.startswith("Non-verbal root"))
        elif key == "Fewer than":
            # filter_min_phrases rejects for two reasons and both must land in
            # this one entry, or the UI's filter card stops summing to
            # total_rejected: the dynamic count label ("Fewer than N preverbal
            # constituent(s)") and the core-argument labels ("Missing preverbal
            # subject" / "... object" / "... subject and object").
            count = sum(
                v for k, v in rejection_counts.items()
                if k.startswith("Fewer than") or k.startswith("Missing preverbal")
            )
            key = "Too few phrases or missing subject/object"
        else:
            count = rejection_counts.get(key, 0)
        filter_order.append({"filter": key, "rejected": count})

    sent_lengths: List[int] = []
    constituent_counts: List[int] = []
    for item in passed_list:
        sent_lengths.append(
            sum(1 for tok in item["sentence"] if isinstance(tok["id"], int))
        )
        constituent_counts.append(len(item["constituents"]))

    constituent_dist: Dict[int, int] = defaultdict(int)
    for c in constituent_counts:
        constituent_dist[c] += 1

    return {
        "total_input": total_input,
        "total_passed": total_passed,
        "total_rejected": total_rejected,
        "pass_rate": round(total_passed / total_input, 4) if total_input else 0.0,
        "rejection_counts": rejection_counts,
        "filter_order": filter_order,
        "avg_sentence_length": round(float(np.mean(sent_lengths)), 2) if sent_lengths else 0.0,
        "avg_constituent_count": round(float(np.mean(constituent_counts)), 2) if constituent_counts else 0.0,
        "constituent_count_distribution": dict(sorted(constituent_dist.items())),
    }
