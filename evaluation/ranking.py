"""
evaluation.ranking
==================
Pairwise ranking evaluation: how well do the scorers' delta features predict
which of the two orderings in a (reference, variant) pair is the one a human
actually wrote?

The pairs table produced by ``generate_variants`` + ``apply_scorers`` is
already the balanced pairwise-transformed dataset of Joachims (2002):
``ML_Label`` alternates 1/0 and every ``Delta_<name>`` column is oriented
``first − second`` by the central diff step, so half the rows present the
reference first and half present it sign-flipped.  Evaluation is therefore a
straight classification task: predict ``ML_Label`` from the delta columns.

The defaults reproduce the evaluation of Ranjan & van Schijndel (2024),
"Does Dependency Locality Predict Non-Canonical Word Order in Hindi?":
10-fold cross-validation (shuffled, seed 42), z-scoring fit on each training
fold only, an lbfgs logistic regression, and accuracy pooled over the
out-of-fold predictions.  Coefficients come from a separate fit on the full
z-scored data with ``C=1e6`` (effectively unregularised, matching R's
``glm()``); their *sign* says which direction the reference is preferred in.

Confidence intervals around the accuracies are currently **suppressed** — see
``INCLUDE_CI`` below.

    from evaluation import evaluate_pairs, evaluate_by_construction

    result = evaluate_pairs(pairs_df)          # every Delta_* column
    result = evaluate_pairs(pairs_df, deltas=["Delta_DL", "Delta_Trigram"])
    per_construction = evaluate_by_construction(pairs_df)   # paper Table 4
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Random baseline for the balanced pairwise task.
CHANCE = 0.5

#: Emit ``ci_low`` / ``ci_high`` alongside each accuracy.  Off for now (the
#: normal-approximation interval assumes independent rows, which pairs from the
#: same sentence are not, so it reads narrower than the truth).  Flip to True to
#: restore the keys — every consumer treats them as optional.
INCLUDE_CI = False


def _cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> float:
    """Pooled out-of-fold accuracy of a z-scored logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler

    if groups is not None:
        if len(np.unique(groups)) < folds:
            raise ValueError(
                f"group_by_sentence needs at least {folds} distinct Sent_IDs "
                f"(got {len(np.unique(groups))})."
            )
        split_iter = GroupKFold(n_splits=folds).split(X, y, groups)
    else:
        split_iter = KFold(n_splits=folds, shuffle=True, random_state=seed).split(X)

    correct = total = 0
    for train_idx, test_idx in split_iter:
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        preds = clf.predict(scaler.transform(X[test_idx]))
        correct += int((preds == y[test_idx]).sum())
        total += len(test_idx)
    return correct / total


def _full_fit_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of an (effectively) unregularised fit on all rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6)
    clf.fit(scaler.transform(X), y)
    return clf.coef_[0]


def _binomial_ci(accuracy: float, n: int) -> tuple:
    """95% normal-approximation confidence interval, clamped to [0, 1].

    Unreferenced while ``INCLUDE_CI`` is False — kept so re-enabling the interval
    is a one-line flip rather than a rewrite."""
    half = 1.96 * math.sqrt(accuracy * (1.0 - accuracy) / n)
    return max(0.0, accuracy - half), min(1.0, accuracy + half)


def _ci_fields(res: dict) -> dict:
    """The confidence-interval keys, or nothing while ``INCLUDE_CI`` is off.

    Splices them back at their original position when the flag is flipped, so
    the key order of a result dict does not depend on the setting."""
    return {k: res[k] for k in ("ci_low", "ci_high") if k in res}


def _direction(coefficient: float) -> str:
    """Plain-English reading of a coefficient sign (delta = first − second,
    label 1 = the reference came first)."""
    if coefficient < 0:
        return "reference preferred when this value is lower"
    return "reference preferred when this value is higher"


def _check_subset(sub: pd.DataFrame, what: str, folds: int) -> None:
    if len(sub) < folds:
        raise ValueError(
            f"Not enough usable rows to evaluate {what}: {len(sub)} rows "
            f"(need at least {folds}, one per fold)."
        )
    if sub["ML_Label"].nunique() < 2:
        raise ValueError(
            f"Cannot evaluate {what}: all usable rows have the same ML_Label."
        )


def evaluate_pairs(
    pairs_df: pd.DataFrame,
    deltas: Optional[List[str]] = None,
    folds: int = 10,
    seed: int = 42,
    labels: Optional[Dict[str, str]] = None,
    group_by_sentence: bool = False,
) -> dict:
    """
    Score each delta feature (and their combination) on pairwise ranking
    accuracy: predicting ``ML_Label`` with 10-fold cross-validation.

    Parameters
    ----------
    pairs_df : DataFrame from ``apply_scorers`` with ``ML_Label`` and one or
               more ``Delta_<name>`` columns.
    deltas   : delta column names to evaluate; ``None`` auto-detects every
               column starting with ``Delta_``.
    folds    : cross-validation folds (default 10, as in the paper).
    seed     : shuffle seed for the fold split (default 42, as in the paper).
    labels   : optional ``{delta_column: display_name}`` (e.g. scorer names)
               echoed into the result for presentation.
    group_by_sentence : when True, split folds by ``Sent_ID`` (GroupKFold) so
               no sentence contributes pairs to both sides of a fold.  Default
               False = plain shuffled KFold, matching the original study.

    Returns
    -------
    dict with ``n_pairs``, ``folds``, ``seed``, ``predictors`` (one entry per
    delta, sorted by accuracy descending) and ``combined`` (all deltas
    together; ``None`` when fewer than two).  Accuracies are 0–1 fractions;
    chance is 0.5.  Rows with a missing value in a delta are dropped for that
    predictor only (``n`` reports what remained).  ``ci_low`` / ``ci_high``
    accompany each accuracy only while the module-level ``INCLUDE_CI`` is True —
    treat them as optional keys.

    Raises
    ------
    ValueError if ``ML_Label`` is missing, no delta columns are found, or a
    predictor has too few usable rows / only one label class.
    """
    if "ML_Label" not in pairs_df.columns:
        raise ValueError("pairs_df has no ML_Label column — run apply_scorers first.")
    if deltas is None:
        deltas = [c for c in pairs_df.columns if c.startswith("Delta_")]
    else:
        missing = [c for c in deltas if c not in pairs_df.columns]
        if missing:
            raise ValueError(f"Delta column(s) not in pairs_df: {missing}")
    if not deltas:
        raise ValueError(
            "No Delta_* columns to evaluate — apply at least one scorer that "
            "declares deltas()."
        )
    if group_by_sentence and "Sent_ID" not in pairs_df.columns:
        raise ValueError("group_by_sentence=True needs a Sent_ID column.")
    labels = labels or {}

    def _evaluate_subset(cols: List[str], what: str) -> dict:
        keep = cols + ["ML_Label"] + (["Sent_ID"] if group_by_sentence else [])
        sub = pairs_df[keep].dropna(subset=cols)
        _check_subset(sub, what, folds)
        X = sub[cols].to_numpy(dtype=float)
        y = sub["ML_Label"].to_numpy(dtype=int)
        groups = sub["Sent_ID"].to_numpy() if group_by_sentence else None
        accuracy = _cv_accuracy(X, y, folds, seed, groups)
        coefs = _full_fit_coefficients(X, y)
        res = {
            "n": int(len(sub)),
            "accuracy": round(float(accuracy), 4),
            "coefficients": {c: round(float(k), 4) for c, k in zip(cols, coefs)},
        }
        if INCLUDE_CI:
            ci_low, ci_high = _binomial_ci(accuracy, len(sub))
            res["ci_low"] = round(float(ci_low), 4)
            res["ci_high"] = round(float(ci_high), 4)
        return res

    predictors = []
    for col in deltas:
        res = _evaluate_subset([col], col)
        coefficient = res["coefficients"][col]
        predictors.append({
            "delta": col,
            "label": labels.get(col, col),
            "n": res["n"],
            "accuracy": res["accuracy"],
            **_ci_fields(res),
            "coefficient": coefficient,
            "direction": _direction(coefficient),
        })
    predictors.sort(key=lambda p: p["accuracy"], reverse=True)

    combined = None
    if len(deltas) >= 2:
        res = _evaluate_subset(list(deltas), "the combined model")
        combined = {
            "deltas": list(deltas),
            "n": res["n"],
            "accuracy": res["accuracy"],
            **_ci_fields(res),
            "coefficients": res["coefficients"],
        }

    return {
        "n_pairs": int(len(pairs_df)),
        "folds": folds,
        "seed": seed,
        "predictors": predictors,
        "combined": combined,
    }


# ---------------------------------------------------------------------------
# Per-construction evaluation (paper Table 4)
# ---------------------------------------------------------------------------

#: The three constructions of Ranjan & van Schijndel (2024), Table 4, as
#: ``(key, label, reference labels)``.  OSV is the union of the other two.
CONSTRUCTIONS = [
    ("OSV", "Object-fronted (Ref OSV vs Var SOV)", frozenset({"DOSV", "IOSV"})),
    ("DOSV", "Direct-object-fronted (Ref DOSV vs Var SDOV)", frozenset({"DOSV"})),
    ("IOSV", "Indirect-object-fronted (Ref IOSV vs Var SIOV)", frozenset({"IOSV"})),
]

_WORD_ORDER_COLUMNS = ("Reference_Word_Order", "Variant_Word_Order")


def construction_mask(pairs_df: pd.DataFrame, reference_labels) -> pd.Series:
    """Boolean mask of the pairs that belong to one construction: the reference
    carries one of *reference_labels* **and** the variant is canonical
    (``SOV``) — a non-canonical original against its canonical counterpart,
    which is what the paper's column headings describe."""
    return (
        pairs_df["Reference_Word_Order"].isin(reference_labels)
        & (pairs_df["Variant_Word_Order"] == "SOV")
    )


def evaluate_by_construction(
    pairs_df: pd.DataFrame,
    deltas: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    min_pairs: int = 1,
    folds: int = 10,
    seed: int = 42,
) -> List[dict]:
    """
    Ranking accuracy per word-order construction — the paper's Table 4 rows.

    Needs the ``Reference_Word_Order`` / ``Variant_Word_Order`` columns from
    ``variants.label_word_orders``.  Each construction's subset is a
    non-canonical reference (``DOSV`` / ``IOSV`` / either, for ``OSV``) paired
    with a canonical ``SOV`` variant, so the DOSV and IOSV subsets are disjoint
    and the OSV subset is exactly their union.  This is stricter than the
    research reproduction's OSV row (which keeps every variant of an
    object-fronted reference); the paper's own pair counts cannot be
    reconstructed from its text, so treat cross-paper comparisons as
    indicative.

    Parameters
    ----------
    pairs_df, deltas, labels, folds, seed : as for :func:`evaluate_pairs`.
    min_pairs : subsets with fewer rows are reported as ``insufficient_pairs``
                rather than evaluated (the web app passes its 200-pair policy).

    Returns
    -------
    A list with one dict per construction, in ``CONSTRUCTIONS`` order:
    ``key``, ``label``, ``n_pairs`` and ``status`` — ``"ok"`` (plus the full
    :func:`evaluate_pairs` result), ``"insufficient_pairs"``, or
    ``"not_evaluable"`` with a ``detail`` message when the subset cannot be
    scored (too few usable rows, or one ``ML_Label`` class).

    Raises
    ------
    ValueError if the word-order columns are missing.
    """
    missing = [c for c in _WORD_ORDER_COLUMNS if c not in pairs_df.columns]
    if missing:
        raise ValueError(
            f"pairs_df lacks {missing} — join variants.label_word_orders() first."
        )
    results = []
    for key, label, ref_labels in CONSTRUCTIONS:
        sub = pairs_df[construction_mask(pairs_df, ref_labels)]
        entry = {"key": key, "label": label, "n_pairs": int(len(sub))}
        if len(sub) < max(min_pairs, 1):
            entry["status"] = "insufficient_pairs"
        else:
            try:
                res = evaluate_pairs(sub, deltas=deltas, folds=folds, seed=seed, labels=labels)
            except ValueError as exc:
                entry["status"] = "not_evaluable"
                entry["detail"] = str(exc)
            else:
                entry["status"] = "ok"
                entry.update(res)
        results.append(entry)
    return results
