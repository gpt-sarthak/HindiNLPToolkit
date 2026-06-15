"""
scoring.is_scorer
=================
Information Status (IS) / givenness scorer.

Reference
---------
Ranjan & van Schijndel (2024),
"Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

The paper scores the ordering of the **subject** and **object** phrases of a
clause by their givenness:

    Given → New   order  → +1
    New   → Given order  → -1
    Given → Given        →  0
    New   → New          →  0

A phrase is GIVEN if its head is a pronoun, or if any of its content-word
lemmas also appear in the preceding context sentence.  The score is computed
separately for the reference order and each variant order, then differenced:

    Delta_IS = IS_Reference - IS_Variant

A positive ``Delta_IS`` means the reference order adheres to given-before-new
more than the variant does (paper footnote 6).

Why this is a context-aware scorer
-----------------------------------
IS needs three things that are not in ``pairs_df``: the dependency parse (to
find subject/object heads, content lemmas and pronoun heads), the preceding
sentence, and the subject/object order *inside each variant*.  The pipeline
supplies the first two through the optional ``context`` argument (see
``scoring.apply_scorers`` and ``helpers.CorpusContext``); the variant
order is reconstructed from the variant surface string.

This scorer never modifies ``variants/generator.py``.

Annotation schemes
-------------------
The subject/object relations, content-word POS tags and pronoun POS tags differ
between UD and Paninian annotation.  ``SCHEME_PRESETS`` holds both; the active
preset is chosen by ``context["scheme"]`` (``"ud"`` or ``"paninian"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from helpers import CorpusContext, recover_permutation
from .base import Scorer


# ---------------------------------------------------------------------------
# Annotation-scheme presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemePreset:
    """Relation / POS sets that identify subject, object, content words and
    pronouns for one annotation scheme."""
    subject_rels: Set[str]
    object_rels: Set[str]
    content_pos: Set[str]
    pronoun_pos: Set[str]


SCHEME_PRESETS: Dict[str, SchemePreset] = {
    # Universal Dependencies (what Stanza emits, and UD Hindi-HDTB).
    "ud": SchemePreset(
        subject_rels={"nsubj", "csubj"},
        object_rels={"obj", "iobj"},
        content_pos={"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"},
        pronoun_pos={"PRON"},
    ),
    # Paninian / SSF tagset (the bundled fact_media_news_ISCNLP.conllu).
    # k1 = karta (subject), k2 = karma (object), k4 = sampradaan (~indirect obj).
    "paninian": SchemePreset(
        subject_rels={"k1"},
        object_rels={"k2", "k4"},
        content_pos={"NN", "NNC", "NNP", "NNPC", "VM", "JJ", "RB", "QC", "QCC"},
        pronoun_pos={"PRP", "PR"},
    ),
}

DEFAULT_SCHEME = "paninian"

_OUTPUT_COLUMNS = ("IS_Reference", "IS_Variant")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _attaching_token(constituent: List[dict], root_id: int) -> Optional[dict]:
    """The token inside *constituent* that attaches directly to *root_id*
    (the phrase head).  None if the constituent has no direct root attachment."""
    for tok in constituent:
        if tok["head"] == root_id:
            return tok
    return None


def _phrase_content_lemmas(constituent: List[dict], preset: SchemePreset) -> Set[str]:
    """Content-word lemmas within a constituent (empty/`_` lemmas excluded)."""
    return {
        tok["lemma"]
        for tok in constituent
        if tok["upos"] in preset.content_pos
        and tok["lemma"]
        and tok["lemma"] != "_"
    }


def _is_given(
    constituent: List[dict],
    head: dict,
    context_lemmas: Set[str],
    preset: SchemePreset,
) -> bool:
    """A phrase is GIVEN if its head is a pronoun, or if its content lemmas
    overlap the context-sentence content lemmas."""
    if head["upos"] in preset.pronoun_pos:
        return True
    return bool(_phrase_content_lemmas(constituent, preset) & context_lemmas)


def _is_score(first_given: bool, second_given: bool) -> int:
    """+1 Given→New, -1 New→Given, 0 otherwise."""
    if first_given and not second_given:
        return 1
    if not first_given and second_given:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Per-sentence precomputation
# ---------------------------------------------------------------------------

@dataclass
class _SentenceIS:
    """Order-independent IS facts for one reference sentence: the givenness of
    its subject and object phrases, the reference IS score, the subject/object
    constituent indices, and the constituents (to recover the variant order)."""
    subj_given: bool
    obj_given: bool
    is_reference: int
    subj_idx: int
    obj_idx: int
    constituents: List[List[dict]] = field(default_factory=list)


def _analyse_sentence(
    root_id: int,
    constituents: List[List[dict]],
    context_lemmas: Set[str],
    preset: SchemePreset,
) -> Optional[_SentenceIS]:
    """Locate the subject and object constituents and compute their givenness.
    Returns None when either a subject or an object cannot be identified
    (matching the reference, which scores 0 in that case)."""
    subj_idx = obj_idx = None
    subj_head = obj_head = None

    for idx, const in enumerate(constituents):
        head = _attaching_token(const, root_id)
        if head is None:
            continue
        if subj_idx is None and head["deprel"] in preset.subject_rels:
            subj_idx, subj_head = idx, head
        elif obj_idx is None and head["deprel"] in preset.object_rels:
            obj_idx, obj_head = idx, head

    if subj_idx is None or obj_idx is None:
        return None

    subj_given = _is_given(constituents[subj_idx], subj_head, context_lemmas, preset)
    obj_given = _is_given(constituents[obj_idx], obj_head, context_lemmas, preset)

    subj_first = subj_idx < obj_idx
    first_given, second_given = (
        (subj_given, obj_given) if subj_first else (obj_given, subj_given)
    )

    return _SentenceIS(
        subj_given=subj_given,
        obj_given=obj_given,
        is_reference=_is_score(first_given, second_given),
        subj_idx=subj_idx,
        obj_idx=obj_idx,
        constituents=constituents,
    )


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class InformationStatusScorer(Scorer):
    name = "information_status"
    description = (
        "Information Status (givenness) of subject vs. object: +1 given-before-new, "
        "-1 new-before-given, 0 otherwise. Delta_IS = IS_Reference - IS_Variant "
        "(positive = reference adheres to given-before-new). Ranjan & van Schijndel 2024."
    )

    def score(self, pairs_df: pd.DataFrame, context: Optional[dict] = None) -> pd.DataFrame:
        df = pairs_df.copy()

        corpus: Optional[CorpusContext] = (context or {}).get("corpus")
        passed = (context or {}).get("passed")
        scheme = (context or {}).get("scheme", DEFAULT_SCHEME)
        preset = SCHEME_PRESETS.get(scheme, SCHEME_PRESETS[DEFAULT_SCHEME])

        # Without the parse + context we cannot compute IS — emit zeros so the
        # columns still exist and downstream code/row-alignment is unaffected.
        if df.empty or corpus is None or not passed:
            for col in _OUTPUT_COLUMNS:
                df[col] = [0] * len(df)
            return df

        # sent_id -> (root_id, constituents)
        parse_by_id: Dict[str, Tuple[int, List[List[dict]]]] = {}
        for item in passed:
            sid = item["sentence"].metadata.get("sent_id", "Unknown_ID")
            parse_by_id.setdefault(sid, (item["root_id"], item["constituents"]))

        # Analyse each sentence once (order-independent givenness facts).
        analysis: Dict[str, Optional[_SentenceIS]] = {}

        is_ref_col: List[int] = []
        is_var_col: List[int] = []

        for sent_id, variant_sentence in zip(df["Sent_ID"], df["Variant_Sentence"]):
            if sent_id not in analysis:
                parse = parse_by_id.get(sent_id)
                if parse is None:
                    analysis[sent_id] = None
                else:
                    root_id, constituents = parse
                    ctx_lemmas = {
                        tok["lemma"]
                        for sent in corpus.previous(sent_id, n=1)
                        for tok in sent
                        if tok["upos"] in preset.content_pos
                        and tok["lemma"]
                        and tok["lemma"] != "_"
                    }
                    analysis[sent_id] = _analyse_sentence(
                        root_id, constituents, ctx_lemmas, preset
                    )

            info = analysis[sent_id]
            if info is None:
                is_ref_col.append(0)
                is_var_col.append(0)
                continue

            is_var = self._variant_score(info, str(variant_sentence))
            is_ref_col.append(info.is_reference)
            is_var_col.append(is_var)

        df["IS_Reference"] = is_ref_col
        df["IS_Variant"] = is_var_col
        return df

    def deltas(self):
        """Delta_IS = ML_Label-oriented difference of the IS scores.

        Note: oriented by the pipeline's flip signal (ref-var on label==1,
        var-ref on label==0), so it matches the paper's ``is_ref - is_var`` on
        reference-first rows and is sign-flipped on the balanced 50% of rows
        presented variant-first."""
        return [(
            "Delta_IS",
            lambda row: row["IS_Reference"],
            lambda row: row["IS_Variant"],
        )]

    @staticmethod
    def _variant_score(info: _SentenceIS, variant_sentence: str) -> int:
        """IS score of one variant: same subject/object givenness as the
        reference, but the surface order is recovered from the variant string
        via the shared ``recover_permutation`` helper."""
        perm = recover_permutation(variant_sentence.split(" "), info.constituents)
        # Whichever of the subject / object constituent appears earlier in the
        # recovered order is "first".  (recover_permutation falls back to the
        # identity order, i.e. the reference order, if it cannot disambiguate.)
        subj_first = perm.index(info.subj_idx) < perm.index(info.obj_idx)

        first_given, second_given = (
            (info.subj_given, info.obj_given)
            if subj_first
            else (info.obj_given, info.subj_given)
        )
        return _is_score(first_given, second_given)
