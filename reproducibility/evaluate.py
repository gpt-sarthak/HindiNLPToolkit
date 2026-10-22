"""
evaluate.py — univariate accuracy and regression coefficients of our word-order
predictors on reproducibility/data/features.csv (built by build_features.py),
compared with Ranjan & van Schijndel (2024), "Does Dependency Locality Predict
Non-canonical Word Order in Hindi?" (rsp/2405.07730v1.pdf).

The pairwise (Joachims) transform is already in the data, done the paper's way:
every pair is one row, ML_Label alternates 1 (reference first) / 0 (variant
first), and each Delta_* is reference - variant on 1-rows and variant -
reference on 0-rows. So the task is: predict ML_Label from the Delta_* columns.
Step 3 checks that this really holds before anything is fitted.

Model, as in the research repo's scripts/train_ranking_model_berkeley.py:
  * accuracy     = 10-fold CV logistic regression (shuffle, seed 42), z-scored per fold
  * coefficients = one logistic regression on the paper's 5 predictors, z-scored, C=1e6
                   (negative = the reference sentence minimises that measure)

Run from the repo root:
    venv\\Scripts\\python.exe reproducibility\\evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
FEATURES_CSV = HERE / "data" / "features.csv"
RESULTS_DIR = HERE / "results"

# The paper's five predictors: label -> (Delta column, raw column prefix).
PAPER_PREDICTORS = {
    "Dependency length":        ("Delta_DL", "DL"),
    "IS score":                 ("Delta_IS", "IS"),
    "PCFG surprisal (DSPS)":    ("Delta_PCFG", "PCFG"),           # the app's berkeley_pcfg
    "Trigram surprisal (Katz)": ("Delta_Trigram", "Trigram"),
    "Adaptive LSTM (Wiki-50M)": ("Delta_AdaptiveWiki50M", "AdaptiveWiki50M"),
}
# Our other LSTM scorers: shown in the accuracy table only.
EXTRA_PREDICTORS = {
    "LSTM (original)":          ("Delta_LSTM", "LSTM"),
    "Adaptive LSTM (original)": ("Delta_Adaptive", "Adaptive"),
    "LSTM (Wiki-50M)":          ("Delta_LSTMWiki50M", "LSTMWiki50M"),
}
ALL_PREDICTORS = {**PAPER_PREDICTORS, **EXTRA_PREDICTORS}

CATEGORIES = ["Full dataset", "OSV vs SOV", "DOSV vs SDOV", "IOSV vs SIOV"]
MIN_PAIRS = 200                     # smaller categories are reported as n/a (the app's floor)

# ─────────────────────────────────────────────────────────────────────────────
# The paper's numbers, copied from rsp/2405.07730v1.pdf for comparison.
#
# Table 1 — pairs per category:  OSV vs SOV 2834 · DOSV vs SDOV 1663 · IOSV vs SIOV 1353
#           (1996 references, 72833 variants in total)
#
# Table 4 — individual prediction accuracy (%), 10-fold CV
#                             OSV vs SOV   DOSV vs SDOV   IOSV vs SIOV
#   IS score                     52.19        53.88          50.92
#   Dependency length            65.24        68.49          58.91
#   PCFG surprisal               67.22        59.05          75.91
#   Trigram surprisal            83.16        78.95          87.29
#   Adaptive LSTM surprisal      83.84        79.98          88.32
#
# Table 2 — regression coefficients, DOSV vs SDOV (N = 1663)
#   Intercept                0.02    Dependency length       -0.58
#   IS score                 0.26    PCFG surprisal           0.52
#   Trigram surprisal       -2.08    Adaptive LSTM surprisal  -1.65
#
# Table 3 — regression coefficients, IOSV vs SIOV (N = 1353)
#   Intercept                0.20    Dependency length        0.15
#   IS score                 0.13    PCFG surprisal          -0.89
#   Trigram surprisal       -3.05    Adaptive LSTM surprisal  -2.83
#
# Not like for like: the paper's PCFG is Modelblocks (5-fold), its trigram is
# SRILM on EMILLE, its LSTM the Neural Complexity Toolkit model. Ours are the
# DSPS Berkeley grammar,
# Katz on Hindi Wikipedia, and the Wiki-50M LSTM. Our PCFG is the app's
# berkeley_pcfg: one DSPS grammar trained on all HDTB trees, so unlike the
# paper's 5-fold PCFG it has seen the test sentences (a leak-free 5-fold
# version is parked in the research repo, leak_free_pcfg/).
# ─────────────────────────────────────────────────────────────────────────────
PAPER_PAIRS = {"OSV vs SOV": 2834, "DOSV vs SDOV": 1663, "IOSV vs SIOV": 1353}
PAPER_ACCURACY = {
    "IS score":                 {"OSV vs SOV": 52.19, "DOSV vs SDOV": 53.88, "IOSV vs SIOV": 50.92},
    "Dependency length":        {"OSV vs SOV": 65.24, "DOSV vs SDOV": 68.49, "IOSV vs SIOV": 58.91},
    "PCFG surprisal (DSPS)":    {"OSV vs SOV": 67.22, "DOSV vs SDOV": 59.05, "IOSV vs SIOV": 75.91},
    "Trigram surprisal (Katz)": {"OSV vs SOV": 83.16, "DOSV vs SDOV": 78.95, "IOSV vs SIOV": 87.29},
    "Adaptive LSTM (Wiki-50M)": {"OSV vs SOV": 83.84, "DOSV vs SDOV": 79.98, "IOSV vs SIOV": 88.32},
}
PAPER_COEFFICIENTS = {
    "Dependency length":        {"DOSV vs SDOV": -0.58, "IOSV vs SIOV": 0.15},
    "IS score":                 {"DOSV vs SDOV": 0.26,  "IOSV vs SIOV": 0.13},
    "PCFG surprisal (DSPS)":    {"DOSV vs SDOV": 0.52,  "IOSV vs SIOV": -0.89},
    "Trigram surprisal (Katz)": {"DOSV vs SDOV": -2.08, "IOSV vs SIOV": -3.05},
    "Adaptive LSTM (Wiki-50M)": {"DOSV vs SDOV": -1.65, "IOSV vs SIOV": -2.83},
}


def main():
    df = step1_load(FEATURES_CSV)
    df = step2_keep_complete_rows(df)
    step3_check_ml_label(df)
    subsets = step4_split_categories(df)
    step5_show_pair_counts(subsets)
    accuracy = step6_univariate_accuracy(subsets)
    coefficients = step7_regression_coefficients(subsets)
    step8_print(accuracy, coefficients)
    step9_save(accuracy, coefficients)


# ── steps ────────────────────────────────────────────────────────────────────

def step1_load(path):
    print(f"Loading {path.relative_to(HERE.parent)} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} pairs, {df['Ref_Index'].nunique():,} reference sentences")
    return df


def step2_keep_complete_rows(df):
    deltas = [delta for delta, _ in ALL_PREDICTORS.values()]
    before = len(df)
    df = df.dropna(subset=deltas).copy()
    print(f"  {len(df):,} pairs have every score (dropped {before - len(df):,})")
    return df


def step3_check_ml_label(df):
    """The data must already be the Joachims transform: a 50/50 ML_Label, and
    every Delta_* = ref - var on 1-rows, var - ref on 0-rows. Stop if not."""
    ones = int((df["ML_Label"] == 1).sum())
    zeros = int((df["ML_Label"] == 0).sum())
    assert ones + zeros == len(df), "ML_Label has values other than 0 and 1"
    # Alternating labels: equal counts, or one apart when the total is odd
    # (the paper's footnote 5).
    assert abs(ones - zeros) <= 1, f"ML_Label is not balanced ({ones:,} ones vs {zeros:,} zeros)"
    share = ones / len(df)
    reference_first = df["ML_Label"] == 1
    for delta, prefix in ALL_PREDICTORS.values():
        ref, var = df[f"{prefix}_Reference"], df[f"{prefix}_Variant"]
        expected = (ref - var).where(reference_first, var - ref)
        worst = (df[delta] - expected).abs().max()
        assert worst < 1e-9, f"{delta} is not oriented by ML_Label (max error {worst})"
    print(f"  ML_Label check passed: share of 1 = {share:.4f}; all {len(ALL_PREDICTORS)} "
          f"deltas are ref-var on 1-rows and var-ref on 0-rows")


def step4_split_categories(df):
    """The paper's categories: a reference with a fronted object against its
    canonical (subject-first) variants. Our variant label SOV = canonical."""
    ref, var = df["Reference_Word_Order"], df["Variant_Word_Order"]
    canonical = var == "SOV"
    return {
        "Full dataset": df,
        "OSV vs SOV":   df[ref.isin(["DOSV", "IOSV"]) & canonical],
        "DOSV vs SDOV": df[(ref == "DOSV") & canonical],
        "IOSV vs SIOV": df[(ref == "IOSV") & canonical],
    }


def step5_show_pair_counts(subsets):
    print("\nPairs per category (paper's count in brackets):")
    for cat, sub in subsets.items():
        paper = f"  ({PAPER_PAIRS[cat]:,})" if cat in PAPER_PAIRS else ""
        print(f"  {cat:<14} {len(sub):>7,}{paper}")


def step6_univariate_accuracy(subsets):
    """10-fold CV accuracy of each predictor on its own, per category."""
    rows = []
    for name, (delta, _) in ALL_PREDICTORS.items():
        for cat, sub in subsets.items():
            rows.append({"predictor": name, "category": cat, "pairs": len(sub),
                         "accuracy": (cv_accuracy(sub[[delta]].to_numpy(), sub["ML_Label"].to_numpy())
                                      if len(sub) >= MIN_PAIRS else np.nan),
                         "paper": PAPER_ACCURACY.get(name, {}).get(cat)})
    return pd.DataFrame(rows)


def step7_regression_coefficients(subsets):
    """One logistic regression on the paper's 5 predictors per category."""
    deltas = [delta for delta, _ in PAPER_PREDICTORS.values()]
    rows = []
    for cat, sub in subsets.items():
        coefs = (fit_coefficients(sub[deltas].to_numpy(), sub["ML_Label"].to_numpy())
                 if len(sub) >= MIN_PAIRS else [np.nan] * len(deltas))
        for name, coef in zip(PAPER_PREDICTORS, coefs):
            rows.append({"predictor": name, "category": cat, "pairs": len(sub),
                         "coefficient": coef, "paper": PAPER_COEFFICIENTS[name].get(cat)})
    return pd.DataFrame(rows)


def step8_print(accuracy, coefficients):
    print("\n" + "=" * 90)
    print("UNIVARIATE ACCURACY (%)                                     random baseline = 50%")
    print("=" * 90)
    print(table(accuracy, "accuracy", ALL_PREDICTORS, "{:6.2f}", "{:6.2f}", split_after=len(PAPER_PREDICTORS)))
    print("\n" + "=" * 90)
    print("REGRESSION COEFFICIENTS (5 predictors in one model)   negative = reference minimises it")
    print("=" * 90)
    print(table(coefficients, "coefficient", PAPER_PREDICTORS, "{:+6.3f}", "{:+5.2f}"))


def step9_save(accuracy, coefficients):
    RESULTS_DIR.mkdir(exist_ok=True)
    accuracy.to_csv(RESULTS_DIR / "univariate_accuracy.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "regression_coefficients.csv", index=False)
    print("\nSaved reproducibility/results/univariate_accuracy.csv and regression_coefficients.csv")


# ── the model (as in the research repo's train_ranking_model_berkeley.py) ────

def cv_accuracy(X, y, n_splits=10, seed=42):
    predictions = np.zeros(len(y))
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        scaler = StandardScaler().fit(X[train])                      # z-score per fold
        model = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train]), y[train])
        predictions[test] = model.predict(scaler.transform(X[test]))
    return (predictions == y).mean() * 100


def fit_coefficients(X, y):
    X = StandardScaler().fit_transform(X)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
    return model.coef_[0]


# ── printing ─────────────────────────────────────────────────────────────────

def table(df, value, predictors, fmt_ours, fmt_paper, split_after=None):
    """Two columns per category: ours, then the paper's value ('-' if none)."""
    header1 = f"{'':<28}" + "".join(f"{c:^16}" for c in CATEGORIES)
    header2 = f"{'':<28}" + "".join(f"{'ours':>7} {'paper':>7} " for _ in CATEGORIES)
    lines = [header1, header2, "-" * len(header1)]
    for i, name in enumerate(predictors):
        if split_after is not None and i == split_after:
            lines.append(f"{'  (our other scorers)':<28}")
        line = f"{name:<28}"
        for cat in CATEGORIES:
            row = df[(df.predictor == name) & (df.category == cat)].iloc[0]
            paper = fmt_paper.format(row.paper) if pd.notna(row.paper) else "-"
            ours = fmt_ours.format(row[value]) if pd.notna(row[value]) else "n/a"
            line += f"{ours:>7} {paper:>7} "
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
