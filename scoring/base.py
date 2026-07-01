"""
scoring.base
============
The contract every scorer plugin must implement.

To add your own scorer
----------------------
1. Create a new file in the ``scoring/`` package, e.g. ``scoring/my_scorer.py``.
2. Subclass :class:`Scorer`, set ``name`` and ``description``, and implement
   ``score()``.  Optionally set the standardized ``trained_on`` / ``built_with``
   / ``notes`` fields — they render as bullets in the scorer list.
3. Done — the web app discovers it automatically and shows it as a checkbox.
   No registration code, no web code.

Example
-------
::

    # scoring/my_scorer.py
    from .base import Scorer

    class MyScorer(Scorer):
        name = "my_scorer"
        description = "One line shown in the web UI."

        def score(self, pairs_df):
            df = pairs_df.copy()
            df["My_Score"] = ...   # one value per (reference, variant) row
            return df

Optional corpus context
------------------------
``score`` may take an extra ``context`` argument if the scorer needs more than
``pairs_df`` (e.g. the dependency parse or the preceding sentence)::

    def score(self, pairs_df, context=None):
        ...

``scoring.apply_scorers`` inspects the signature and passes ``context`` only to
scorers that declare it, so single-argument scorers keep working unchanged.
The pipeline supplies a read-only dict conventionally holding:

    corpus  : a ``helpers.CorpusContext`` (preceding-sentence lookup)
    passed  : the filter output (``sentence`` / ``root_id`` / ``constituents``)
    scheme  : annotation scheme, ``"ud"`` or ``"paninian"``

Treat ``context`` as read-only: it may be shared across rows and (for the web
app) across concurrent jobs.
"""

from __future__ import annotations

import pandas as pd


class Scorer:
    """
    Base class for reference/variant pair scorers.

    Attributes
    ----------
    name        : unique machine name (used in API requests and the registry).
    description : one-line human description (shown in the web UI).

    Optional standardized metadata (rendered as bullets in the web UI scorer
    list and surfaced by ``/api/plugins``).  Leave any of them ``""`` to omit
    that bullet; an empty ``trained_on`` reads as "not trained / deterministic":

    trained_on  : the data the model was trained on.
    built_with  : the underlying model or method used to build it.
    notes       : anything else worth stating (smoothing, units, caveats).

    needs_previous_sentence : True if the scorer needs the *preceding* sentence
        to be meaningful (e.g. adaptation on context, or givenness vs. the prior
        sentence).  The web app skips such scorers in single-sentence runs when
        no context sentence is supplied.
    """

    name: str = ""
    description: str = ""
    trained_on: str = ""
    built_with: str = ""
    notes: str = ""
    needs_previous_sentence: bool = False

    def score(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one or more score columns to the reference/variant pairs table.

        Parameters
        ----------
        pairs_df : DataFrame from ``generate_variants()`` with columns
                   Sent_ID | Variant_ID | Reference_Sentence | Variant_Sentence |
                   Ref_Features | Var_Features | Diff_Features | ML_Label
                   (plus any columns added by scorers that ran before this one).

        Returns
        -------
        DataFrame — the same rows, with new column(s) added.  Do not drop or
        reorder rows; downstream code relies on row alignment.
        """
        raise NotImplementedError

    def deltas(self):
        """
        Declare the per-feature diffs this scorer wants computed at the end of
        the pipeline from ``ML_Label`` (the *flip signal*).

        Return a list of ``(delta_column_name, ref_fn, var_fn)`` tuples, where
        ``ref_fn``/``var_fn`` take a DataFrame row and return the reference-side
        and variant-side scalar score.  ``apply_scorers`` then writes::

            delta = ref - var   if row["ML_Label"] == 1   (reference first)
            delta = var - ref   if row["ML_Label"] == 0   (flipped)

        Scorers should emit raw ``*_Reference`` / ``*_Variant`` (true-role)
        columns in ``score`` and leave the diff orientation to this step, so
        every feature is differenced consistently.  Default: no deltas.
        """
        return []
