"""
scoring.example_scorer
======================
Template for writing your own scorer — copy this file, rename it, uncomment the
class, and fill in ``score``.

Nothing here is registered: the example class is intentionally left commented
out so it does not add a column to every run.  For real, working scorers to
read alongside this template, see:

    scoring/dl_scorer.py   — dependency_length  (5-element DL feature vectors)
    scoring/is_scorer.py   — information_status  (givenness, needs corpus context)

A scorer adds one or more columns to ``pairs_df`` (the ``generate_variants``
output) and must never drop or reorder rows.  If it needs the parse or the
preceding sentence, it declares an optional ``context`` parameter (see
``scoring.base`` and ``helpers``).  The per-feature diff is computed
centrally from ``ML_Label`` at the end of the pipeline — declare it via
``deltas`` rather than computing it yourself.

Example
-------
::

    import ast
    from .base import Scorer

    def _as_list(value):
        # Feature columns hold real lists in memory but strings after a CSV
        # round-trip; accept both.
        return ast.literal_eval(value) if isinstance(value, str) else list(value)

    class MyScorer(Scorer):
        name = "my_scorer"                  # shown as a checkbox in the UI
        description = "One line shown in the web UI."

        def score(self, pairs_df):
            df = pairs_df.copy()
            df["My_Ref_Score"] = ...        # one value per (reference, variant) row
            df["My_Var_Score"] = ...
            return df

        def deltas(self):
            # Delta_My = ML_Label-oriented (ref - var) of the two scores.
            return [("Delta_My",
                     lambda row: row["My_Ref_Score"],
                     lambda row: row["My_Var_Score"])]
"""

from __future__ import annotations
