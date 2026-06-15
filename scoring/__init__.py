"""
scoring
=======
Plugin package for scoring (reference, variant) sentence pairs.

Drop a module in this package that subclasses ``scoring.base.Scorer`` and it
is discovered automatically — see ``scoring/base.py`` for the contract and
``scoring/example_scorer.py`` for a working example.

    from scoring import get_scorers, apply_scorers

Scorers that need the surrounding corpus (e.g. the Information Status scorer)
declare an optional ``context`` parameter on their ``score`` method; the
``helpers`` package provides the reusable ``CorpusContext`` helper for
resolving preceding sentences.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional

import pandas as pd

from .base import Scorer
# Re-exported for backward compatibility; the implementation now lives in the
# top-level `helpers` package.
from helpers import CorpusContext, build_corpus_context

_registry: Dict[str, Scorer] = {}
_discovered = False


def get_scorers() -> Dict[str, Scorer]:
    """
    Return ``{name: scorer_instance}`` for every Scorer subclass found in the
    ``scoring`` package.  Modules are imported once and cached.
    """
    global _discovered
    if not _discovered:
        for modinfo in pkgutil.iter_modules(__path__):
            if modinfo.name == "base":
                continue
            module = importlib.import_module(f"{__name__}.{modinfo.name}")
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, Scorer)
                    and cls is not Scorer
                    and cls.__module__ == module.__name__
                ):
                    instance = cls()
                    if not instance.name:
                        raise ValueError(
                            f"Scorer {cls.__qualname__} in {module.__name__} "
                            "has no 'name' attribute set."
                        )
                    if instance.name in _registry:
                        raise ValueError(
                            f"Duplicate scorer name '{instance.name}' "
                            f"({cls.__qualname__} in {module.__name__})."
                        )
                    _registry[instance.name] = instance
        _discovered = True
    return _registry


def _accepts_context(scorer: Scorer) -> bool:
    """True if a scorer's ``score`` method takes a ``context`` argument (or
    ``**kwargs``).  Lets us pass context only to scorers that want it, keeping
    older single-argument scorers working unchanged."""
    params = inspect.signature(scorer.score).parameters
    return "context" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def apply_scorers(
    pairs_df: pd.DataFrame,
    names: List[str],
    context: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run the named scorers in order over *pairs_df* and return the result.

    Parameters
    ----------
    pairs_df : DataFrame from ``generate_variants``.
    names    : scorer names to run, in order.
    context  : optional read-only dict of corpus context passed to scorers that
               accept a ``context`` argument.  Conventionally holds
               ``corpus`` (a :class:`~helpers.CorpusContext`),
               ``passed`` (the filter output), and ``scheme`` (``"ud"`` /
               ``"paninian"``).  Passed as a call argument — never stored on the
               shared scorer instance — so concurrent jobs stay independent.

    Raises KeyError if a name is not in the registry.
    """
    scorers = get_scorers()
    ran: List[Scorer] = []
    for name in names:
        if name not in scorers:
            raise KeyError(
                f"Unknown scorer '{name}'. Available: {sorted(scorers)}"
            )
        scorer = scorers[name]
        if _accepts_context(scorer):
            pairs_df = scorer.score(pairs_df, context=context)
        else:
            pairs_df = scorer.score(pairs_df)
        ran.append(scorer)

    # Central diff step: compute each scorer's declared deltas from ML_Label,
    # so every feature is differenced with the same (first - second) orientation.
    pairs_df = _apply_deltas(pairs_df, ran)
    return pairs_df


def _apply_deltas(pairs_df: pd.DataFrame, scorers: List[Scorer]) -> pd.DataFrame:
    """Write ``Delta_<name>`` columns declared by *scorers*, oriented by
    ``ML_Label`` (ref-var when 1, var-ref when 0)."""
    if pairs_df.empty:
        return pairs_df
    specs = [(n, rf, vf) for s in scorers for (n, rf, vf) in s.deltas()]
    if not specs:
        return pairs_df
    has_label = "ML_Label" in pairs_df.columns
    records = pairs_df.to_dict("records")  # faster than iterrows for row callables
    for delta_name, ref_fn, var_fn in specs:
        values = []
        for row in records:
            ref_val, var_val = ref_fn(row), var_fn(row)
            flipped = has_label and int(row["ML_Label"]) == 0
            values.append(var_val - ref_val if flipped else ref_val - var_val)
        pairs_df[delta_name] = values
    return pairs_df


__all__ = [
    "Scorer",
    "get_scorers",
    "apply_scorers",
    "CorpusContext",
    "build_corpus_context",
]
