"""
variants
========
Generates grammatically valid preverbal constituent permutations paired with
the corpus reference order, for dependency-length ML research.

    from variants import generate_variants, label_word_orders

``generate_variants`` emits surface pairs only; ``label_word_orders`` adds the
paper's word-order construction labels (``SOV`` / ``DOSV`` / ``IOSV``) for the
reference and the variant of every row.
"""

from .generator import generate_variants
from .word_order import WORD_ORDER_COLUMNS, label_word_orders

__all__ = ["generate_variants", "label_word_orders", "WORD_ORDER_COLUMNS"]
