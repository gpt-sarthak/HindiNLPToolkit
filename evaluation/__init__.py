"""
evaluation
==========
Pairwise ranking evaluation of scorer delta features: given the pairs table
from ``generate_variants`` + ``apply_scorers`` (``ML_Label`` + ``Delta_*``
columns), measure how often each feature — alone and combined — picks the
human-written ordering over the reordered variant (10-fold CV logistic
regression; chance = 50%).  See ``evaluation/ranking.py`` for the method.

    from evaluation import evaluate_pairs
"""

from .ranking import evaluate_pairs

__all__ = ["evaluate_pairs"]
