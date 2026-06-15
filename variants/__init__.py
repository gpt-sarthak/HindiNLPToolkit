"""
variants
========
Generates grammatically valid preverbal constituent permutations paired with
the corpus reference order, for dependency-length ML research.

    from variants import generate_variants
"""

from .generator import generate_variants

__all__ = ["generate_variants"]
