# Contributing a Scorer

This is a group project. Everyone contributes by adding their own **scorer** —
a small module that adds feature column(s) to the reference/variant sentence
pairs. This guide covers the two things you need: **how the `scoring/` folder
works**, and the **git branch workflow** to implement your scorer from the
original repo and get it merged.

You only ever touch the `scoring/` folder. You never edit the web app or the
generation/filtering engine. Full API details live in [`DOCS.md`](DOCS.md) —
this file is the quick start.

---

## 1. Get set up

```powershell
git clone https://github.com/gpt-sarthak/HindiNLPToolkit.git
cd HindiNLPToolkit
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the web app locally to see your scorer as a checkbox
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

> The first time you parse a `.txt`/raw-string input, Stanza downloads the Hindi
> model to `~/stanza_resources/` (slow once, cached after). Pre-annotated
> `.conllu` uploads don't need it.

---

## 2. Branch workflow (implement from the original)

Always start from an up-to-date `main`, and do your work on a dedicated branch —
one scorer per branch/PR. Keep `main` clean.

```powershell
git checkout main
git pull                                   # get everyone else's merged work

git checkout -b scorer/<your-scorer-name>  # e.g. scorer/surprisal

# ... add your file in scoring/ and test it (sections 3-5) ...

git add scoring/<your_scorer>.py
git commit -m "Add <your-scorer-name> scorer"
git push -u origin scorer/<your-scorer-name>
```

Then open a **Pull Request** into `main` on GitHub and have another teammate
review and merge it.

- **Collaborators** (added to the repo): clone and branch directly, as above.
- **Not a collaborator?** Fork the repo on GitHub, then clone your fork, branch,
  push to your fork, and open a PR from the fork into the original `main`.

---

## 3. Write the scorer

Create one file in `scoring/`, e.g. `scoring/surprisal_scorer.py`. Subclass
`Scorer`, set `name` + `description`, and implement `score`. That's it — the web
app **auto-discovers** it on restart and shows it as a checkbox. No registration,
no web code.

```python
from .base import Scorer

class SurprisalScorer(Scorer):
    name = "surprisal"                      # machine name + checkbox label
    description = "One line shown in the web UI."

    def score(self, pairs_df):
        df = pairs_df.copy()
        df["Surprisal_Reference"] = ...     # reference-side value (true role)
        df["Surprisal_Variant"]   = ...     # variant-side value  (true role)
        return df

    def deltas(self):
        # The pipeline computes Delta_<name> for you from ML_Label — see below.
        return [("Delta_Surprisal",
                 lambda row: row["Surprisal_Reference"],
                 lambda row: row["Surprisal_Variant"])]
```

What `score` receives — `pairs_df`, the reference/variant pairs table. These
columns are **always present**:

| Column | Meaning |
| --- | --- |
| `Sent_ID` | id of the source sentence (use it to look up corpus context) |
| `Variant_ID` | `{Sent_ID}_v{n}` |
| `ML_Label` | flip signal: `1` = reference shown first, `0` = variant shown first |
| `Reference_Sentence` | surface text of the original order |
| `Variant_Sentence` | surface text of the reordered variant |

Rules — **do not break these:**

- **Add columns only. Never drop or reorder rows** — downstream code relies on
  row alignment.
- **Store true-role values:** `*_Reference` for the reference order,
  `*_Variant` for the variant order.
- **Don't compute the difference yourself.** Declare it in `deltas()`. After all
  selected scorers run, `apply_scorers` computes every `Delta_<name>` from the
  `ML_Label` flip signal, so all features are differenced consistently:

  ```
  Delta_<name> = ref - var   if ML_Label == 1   (reference shown first)
  Delta_<name> = var - ref   if ML_Label == 0   (variant shown first)
  ```

Read these before you start: `scoring/example_scorer.py` (copy-paste template),
`scoring/dl_scorer.py` and `scoring/is_scorer.py` (the two built-in scorers).

---

## 4. Using previous-sentence context (and ML models)

If your scorer needs more than the surface strings — the dependency parse, or
the **preceding sentence(s)** — add an optional `context` parameter to `score`.
`apply_scorers` passes `context` only to scorers that declare it, so simpler
scorers are unaffected.

```python
def score(self, pairs_df, context=None):
    df = pairs_df.copy()
    corpus = (context or {}).get("corpus")   # helpers.CorpusContext | None

    ref_scores, var_scores = [], []
    for sent_id, ref_text, var_text in zip(
        df["Sent_ID"], df["Reference_Sentence"], df["Variant_Sentence"]
    ):
        # The 1-2 sentences before this one, oldest first ("" if none/no corpus).
        prev = corpus.previous_text(sent_id, n=2) if corpus else ""
        ref_scores.append(self.run_model(ref_text, prev))   # your model
        var_scores.append(self.run_model(var_text, prev))
    df["MyModel_Reference"] = ref_scores
    df["MyModel_Variant"]   = var_scores
    return df
```

The `context` dict holds:

- `context["corpus"]` — a `helpers.CorpusContext`. Use
  `corpus.previous_text(sent_id, n=2)` for the previous **text**, or
  `corpus.previous(sent_id, n=2)` for the previous sentences as parsed tokens.
  `n=1` for a single previous sentence; returns `""`/`[]` at the corpus start.
- `context["passed"]` — the filter output (parse / constituents).
- `context["scheme"]` — `"ud"` or `"paninian"`.

**ML-model scorers** fit this directly: your model takes one sentence (optionally
plus its previous-sentence context) and returns a number. Score the reference
order and the variant order **separately** into `*_Reference` / `*_Variant`, and
declare the `deltas()` — the pipeline handles the rest.

- **Load the model once**, not per row (e.g. a `functools.cached_property` on the
  instance), so app startup isn't blocked and you don't reload every call.
- **Treat `context` as read-only** — it is shared across rows and across
  concurrent web jobs (2 workers). Don't mutate it or stash it on `self`.
- **Keep your model weights and your own loader module out of this repo.** The
  scorer file goes in `scoring/`; large weight files should be gitignored.

---

## 5. Test it

Restart the web app — your scorer appears as a checkbox. Or check from Python:

```python
from scoring import get_scorers, apply_scorers
print(sorted(get_scorers()))                      # your name should be listed

out = apply_scorers(pairs_df, ["<your-name>"], context=context)
# confirm your columns + Delta_<name> appear and the row count is unchanged
```

---

## 6. Conventions & don'ts

- **Don't import `webapp` from `scoring`** — dependencies point one way only
  (`webapp` → logic packages, never the reverse).
- **Don't mix annotation schemes** in one run — Paninian (`k1`/`k2`/`vmod`, root
  POS `VM`/`VAUX`) and UD (`nsubj`/`obj`/`obl`, copular roots) have zero deprel
  overlap.
- **Don't commit** `venv/`, `__pycache__/`, model weights, or large corpora —
  `.gitignore` already covers the common cases; add your weight files if needed.
- Reusable, scorer-agnostic primitives (previous-sentence lookup, rebuilding a
  variant's tree) live in `helpers/` — check there before writing your own.

Full API reference: [`DOCS.md`](DOCS.md).
