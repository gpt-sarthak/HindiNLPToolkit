"""
evaluation
==========
Pairwise ranking evaluation of scorer delta features: given the pairs table
from ``generate_variants`` + ``apply_scorers`` (``ML_Label`` + ``Delta_*``
columns), measure how often each feature — alone and combined — picks the
human-written ordering over the reordered variant (10-fold CV logistic
regression; chance = 50%).  See ``evaluation/ranking.py`` for the method.

    from evaluation import evaluate_pairs, evaluate_by_construction

``evaluate_by_construction`` repeats the measurement per word-order
construction (OSV / DOSV / IOSV references against canonical variants — the
paper's Table 4), using the ``Reference_Word_Order`` / ``Variant_Word_Order``
columns from ``variants.label_word_orders``.
"""

from .ranking import CONSTRUCTIONS, construction_mask, evaluate_by_construction, evaluate_pairs

__all__ = ["evaluate_pairs", "evaluate_by_construction", "construction_mask", "CONSTRUCTIONS"]
