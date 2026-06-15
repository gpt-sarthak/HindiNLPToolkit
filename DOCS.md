# Hindi NLP Toolkit — Documentation

A Python library for filtering Hindi CoNLL-U sentences and generating
preverbal constituent order variants for dependency-length research.

---

## Setup

```powershell
# Activate the virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install -r requirements.txt
```

---

## Importing

```python
from stanza_parser import load_input                      # unified loader

from filtering import filter_sentences, summarize         # combined pipeline
from filtering import (                                   # individual filters
    filter_questions,
    filter_negatives,
    filter_ghost_ids,
    filter_non_projective,
    filter_bad_root,
    filter_punct_constituents,
    filter_min_phrases,
)

from variants import generate_variants
```

---

## Quick start

```python
from stanza_parser import load_input
from filtering import filter_sentences, summarize
from variants import generate_variants

# 1. Load
sentences = load_input("fact_media_news_ISCNLP.conllu")

# 2. Filter (full pipeline)
passed, rejected_df, passed_df = filter_sentences(sentences, output_dir="output/")

# 3. Generate variants (surface pairs only — no features)
pairs_df = generate_variants(passed, output_dir="output/")

# 4. Score (features live in the scoring package)
from scoring import apply_scorers, build_corpus_context
pairs_df = apply_scorers(
    pairs_df,
    ["dependency_length", "information_status"],
    context={"corpus": build_corpus_context(sentences), "passed": passed, "scheme": "paninian"},
)

# 5. Summarize
print(summarize(passed, rejected_df))
```

---

## Function Reference

### `load_input(source)`

Loads sentences from a file or a raw Hindi string.

| Input | Behaviour |
|---|---|
| Path ending in `.conllu` | Parsed directly with the `conllu` library. No model needed. |
| Path ending in `.txt` | Each non-empty line parsed as one sentence via Stanza (Hindi). |
| Raw Hindi string | Parsed via Stanza (Hindi). Model downloaded automatically on first use. |

**Returns** `List[TokenList]` — list of `conllu` TokenList objects.

**Raises**
- `FileNotFoundError` — if a `.conllu` or `.txt` path does not exist.
- `ValueError` — if the source is empty or contains no parseable content.
- `ImportError` — if Stanza is not installed and is needed.

```python
sents = load_input("corpus.conllu")
sents = load_input("sentences.txt")          # one sentence per line
sents = load_input("राम घर जाता है।")       # raw Hindi string
```

---

### `filter_sentences(sentences, allowed_root_pos=None, min_phrases=2, output_dir=None)`

Applies all seven linguistic filters in sequence. A sentence must pass **all**
filters to be included. Internally calls the seven individual filter functions
in order — no logic is duplicated.

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `sentences` | — | `List[TokenList]` from `load_input()`. |
| `allowed_root_pos` | `None` | POS tags accepted on the dependency root (filter 5). `None` uses `["VERB", "AUX", "VM", "VAUX"]`. Extend for UD corpora with copular constructions, e.g. `["VERB", "AUX", "NOUN", "ADJ"]`. |
| `min_phrases` | `2` | Minimum preverbal constituents required (filter 7). Must be ≥ 1. |
| `output_dir` | `None` | If given, writes `passed_sentences.csv` and `rejected_sentences.csv` to this directory. |

#### Filter pipeline

| # | Function | What it checks | Why |
|---|---|---|---|
| 1 | `filter_questions` | `?` in text, `WQ` POS tag, or `PronType=Int` feature | Questions have different word-order constraints |
| 2 | `filter_negatives` | `NEG` POS tag or `Polarity=Neg` feature | Negation interacts with constituent order outside this model's scope |
| 3 | `filter_ghost_ids` | Any token with a non-integer ID (e.g. `1.1`) | Empty nodes break position arithmetic |
| 4 | `filter_non_projective` | Any pair of crossing dependency arcs | Non-projective structures cannot be represented by block permutations |
| 5 | `filter_bad_root` | Missing root, or root POS not in `allowed_root_pos` | Preverbal constituent extraction is only defined for verb-headed clauses |
| 6 | `filter_punct_constituents` | Any preverbal constituent whose head attachment to root has `deprel == "punct"` | A bare comma or punctuation mark treated as a free-floating block produces broken variants in every permutation |
| 7 | `filter_min_phrases` | Fewer than `min_phrases` preverbal constituents | Need ≥ 2 to produce any non-trivial permutation |

#### Returns

| Name | Type | Description |
|---|---|---|
| `passed_list` | `List[Dict]` | Dicts with keys `sentence`, `root_id`, `constituents`. Direct input to `generate_variants()`. |
| `rejected_df` | `DataFrame` | `Sent_ID`, `Sentence`, `Reason` — one row per rejected sentence. |
| `passed_df` | `DataFrame` | Rich feature table; see columns below. |

**`passed_df` columns**

| Column | Description |
|---|---|
| `Sent_ID` | Sentence identifier from CoNLL-U metadata |
| `Sentence` | Surface text |
| `Root_ID` | Token ID of the verbal root |
| `Phrase_Count` | Number of preverbal constituents |
| `Character_Length` | Character count of surface text |
| `Sentence_Length` | Token count (integer IDs only) |
| `Constituent_Lengths` | List of token counts per constituent |
| `Deprel_Tags` | List of dependency relation labels per constituent |
| `Grammatical_Pairs` | Adjacent `(deprel_A, deprel_B)` pairs in reference order |

**Files saved** (when `output_dir` is given)
- `passed_sentences.csv`
- `rejected_sentences.csv`

```python
passed, rejected_df, passed_df = filter_sentences(sents, output_dir="output/")

# UD corpus with copular constructions (ADJ/NOUN roots)
passed, rejected_df, passed_df = filter_sentences(
    sents,
    allowed_root_pos=["VERB", "AUX", "NOUN", "ADJ", "PROPN"],
    output_dir="output/",
)

# Require at least 3 preverbal phrases
passed, rejected_df, passed_df = filter_sentences(sents, min_phrases=3)
```

---

### Individual filter functions

Each filter can be called independently, which is useful when you want to apply
only a subset of filters, inspect intermediate results, or chain them in a
custom order.

**Type progression through the pipeline**

Filters 1–4 accept and return `List[TokenList]`. Filter 5 (`filter_bad_root`)
extracts preverbal constituents and switches the type to `List[Dict]`. Filters
6–7 accept either type (they auto-detect from the first element).

Every individual filter raises `ValueError` on empty input and returns
`(passed, rejected_df)` where `rejected_df` always has columns
`Sent_ID | Sentence | Reason`.

---

#### `filter_questions(sentences, check_punct=True, check_pos=True, check_features=True)`

Removes interrogative sentences. Three independent signals are checked; a
sentence is rejected if any enabled signal fires.

| Parameter | Default | Signal checked |
|---|---|---|
| `check_punct` | `True` | `?` anywhere in the surface text |
| `check_pos` | `True` | Any token with `upos == "WQ"` (Paninian tagset) |
| `check_features` | `True` | Any token with `PronType=Int` morphological feature (UD convention) |

**Returns** `(List[TokenList], DataFrame)`

```python
passed, rej = filter_questions(sentences)

# Skip the punctuation check (e.g. corpus uses ？ instead of ?)
passed, rej = filter_questions(sentences, check_punct=False)

# Disable all signals — passes everything (useful for testing)
passed, rej = filter_questions(sentences, check_punct=False, check_pos=False, check_features=False)
```

---

#### `filter_negatives(sentences, check_pos=True, check_features=True)`

Removes sentences that contain negation.

| Parameter | Default | Signal checked |
|---|---|---|
| `check_pos` | `True` | Any token with `upos == "NEG"` (Paninian tagset; fires only on ISCNLP-style corpora) |
| `check_features` | `True` | Any token with `Polarity=Neg` morphological feature (UD convention; catches नहीं, मत, etc.) |

**Returns** `(List[TokenList], DataFrame)`

```python
passed, rej = filter_negatives(sentences)
```

> **Note:** On UD-annotated corpora (e.g. Hindi-HDTB) the `check_pos` signal
> fires zero times because UD does not use a `NEG` POS tag. All negation is
> caught by `check_features`. Both signals can safely be left on for any corpus.

---

#### `filter_ghost_ids(sentences)`

Removes sentences that contain non-integer (ghost/empty) token IDs such as
`1.1` used in enhanced dependency graphs. These break all integer-position
arithmetic used downstream.

**Returns** `(List[TokenList], DataFrame)`

```python
passed, rej = filter_ghost_ids(sentences)
```

---

#### `filter_non_projective(sentences)`

Removes sentences whose dependency tree contains crossing arcs. Non-projective
structures cannot be faithfully represented by constituent-block permutations.

**Returns** `(List[TokenList], DataFrame)`

```python
passed, rej = filter_non_projective(sentences)
```

> **Corpus note:** UD-annotated Hindi (HDTB) has a non-projectivity rate of
> ~12%, roughly 2.4× higher than Paninian-annotated corpora (~5%). This is a
> genuine annotation-scheme difference, not a pipeline issue.

---

#### `filter_bad_root(sentences, allowed_pos=None)`

Removes sentences whose dependency root is missing or carries a POS tag not in
`allowed_pos`, then extracts preverbal constituents for sentences that pass.
This is the point where the output type changes from `List[TokenList]` to
`List[Dict]`.

| Parameter | Default | Description |
|---|---|---|
| `allowed_pos` | `None` | POS tags accepted on the root. `None` → `["VERB", "AUX", "VM", "VAUX"]`. |

**Returns** `(List[Dict], DataFrame)`

Each dict in the passed list has:
- `sentence` — the original `TokenList`
- `root_id` — integer token ID of the verbal root
- `constituents` — `List[List[Dict]]`, one sub-list per preverbal constituent

The `Reason` field in the rejection DataFrame includes the actual POS found,
e.g. `"Bad Root: Root POS is 'NN', expected VERB/AUX/VM/VAUX"`.

```python
passed, rej = filter_bad_root(sentences)

# UD corpus — allow nominal/adjectival roots from copular constructions
passed, rej = filter_bad_root(sentences, allowed_pos=["VERB", "AUX", "NOUN", "ADJ", "PROPN"])
```

> **UD copular constructions:** In UD treebanks the predicate (ADJ or NOUN) is
> the syntactic root when the copula `है` is annotated as `AUX/cop`. These
> sentences are rejected by the default `allowed_pos`. Extend the list when
> working with UD data if copular sentences are linguistically relevant to your
> research.

---

#### `filter_punct_constituents(sentences_or_items, allowed_pos=None)`

Removes sentences where any preverbal constituent's direct attachment to the
root carries `deprel == "punct"`. In practice this is always a lone comma or
quotation mark from a discourse-connector opening like `"लेकिन,"` or
`"बहरहाल,"`, where UD annotation gives the connector and the comma separate
head attachments to the verb.

**Why this matters:** because constituent permutation treats every constituent
as a free-floating block, a bare comma block moves freely to the middle or
start of every variant, producing output that is ungrammatical in all
permutations:

```
Reference : लेकिन , उसकी दृढ़ इच्छा के आगे उन्हें झुकना पड़ा ।   ✓
Variant A : उसकी दृढ़ इच्छा के आगे , लेकिन उन्हें झुकना पड़ा ।   ✗
Variant B : , उसकी दृढ़ इच्छा के आगे लेकिन उन्हें झुकना पड़ा ।   ✗
            (all 23 permutations of this sentence are broken)
```

| Parameter | Default | Description |
|---|---|---|
| `allowed_pos` | `None` | Only used when input is `List[TokenList]`. POS tags accepted on the root. |

**Accepts** `List[TokenList]` or `List[Dict]` (output of `filter_bad_root`).  
**Returns** `(List[Dict], DataFrame)`

```python
# Chained after filter_bad_root (typical use)
items, _ = filter_bad_root(sentences)
passed, rej = filter_punct_constituents(items)

# Independent use on raw sentences
passed, rej = filter_punct_constituents(sentences)
```

> **Corpus note:** This filter fires only on UD-annotated corpora. The ISCNLP
> Paninian corpus has zero such sentences because Paninian annotation does not
> attach punctuation as a direct preverbal arc to the root.

---

#### `filter_min_phrases(sentences_or_items, min_phrases=2, allowed_pos=None)`

Removes sentences that have fewer than `min_phrases` preverbal constituents.
At least two are needed to produce any non-trivial permutation.

| Parameter | Default | Description |
|---|---|---|
| `min_phrases` | `2` | Minimum number of preverbal constituents required. Must be ≥ 1. |
| `allowed_pos` | `None` | Only used when input is `List[TokenList]`. POS tags accepted on the root. |

**Accepts** `List[TokenList]` or `List[Dict]` (output of `filter_bad_root` /
`filter_punct_constituents`). When given `List[Dict]`, pre-computed
constituents are used directly — no re-extraction.  
**Returns** `(List[Dict], DataFrame)`

```python
# Typical: chained after filter_punct_constituents
items, _ = filter_bad_root(sentences)
items, _ = filter_punct_constituents(items)
passed, rej = filter_min_phrases(items, min_phrases=2)

# Stricter threshold
passed, rej = filter_min_phrases(items, min_phrases=3)

# Independent use on raw sentences
passed, rej = filter_min_phrases(sentences)
```

---

#### Chaining individual filters

You can compose any subset of filters manually. The output of each filter is a
valid input for the next.

```python
from stanza_parser import load_input
from filtering import (
    filter_questions, filter_negatives, filter_ghost_ids,
    filter_non_projective, filter_bad_root,
    filter_punct_constituents, filter_min_phrases,
)
import pandas as pd

sentences = load_input("corpus.conllu")

p1, r1 = filter_questions(sentences)
p2, r2 = filter_negatives(p1)
p3, r3 = filter_ghost_ids(p2)
p4, r4 = filter_non_projective(p3)
p5, r5 = filter_bad_root(p4, allowed_pos=["VERB", "AUX", "NOUN", "ADJ"])
p6, r6 = filter_punct_constituents(p5)
passed, r7 = filter_min_phrases(p6, min_phrases=2)

all_rejected = pd.concat([r1, r2, r3, r4, r5, r6, r7], ignore_index=True)
```

---

### `generate_variants(passed_list, valid_deprel_pairs=None, max_variants=99, seed=42, output_dir=None)`

For each sentence, generates permutations of its preverbal constituents and
pairs them with the corpus (reference) order. **Surface only** — it produces the
sentence pairs and the ML pairing label, but computes **no features**. Feature
extraction (dependency length, information status, …) lives in the `scoring`
package.

#### How variants are generated

1. The preverbal constituents are taken from `passed_list` (already computed by `filter_sentences` or the individual filter chain).
2. All permutations of those constituents are enumerated (or sampled for long sentences).
3. Each permutation is checked against `valid_deprel_pairs` — only permutations where every adjacent dependency-relation pair has been observed in the corpus are kept.
4. Up to `max_variants` valid permutations are selected per sentence.
5. Each (reference, variant) pair is emitted as one row with an alternating `ML_Label` — the *flip signal*: `1` = presented reference-first, `0` = flipped (variant-first). This guarantees an exactly balanced 50/50 split and is what every feature scorer later uses to orient its diff.

#### Grammar filter (`valid_deprel_pairs`)

The filter keeps only permutations that produce plausible adjacent deprel bigrams.
- If `None` (default): the set is built from all deprel adjacencies observed across `passed_list`.
- Pass a custom set to restrict or relax the filter, e.g. from a larger external corpus.

> **Cross-corpus warning:** The `valid_deprel_pairs` built from one corpus is
> incompatible with sentences from another if their annotation schemes differ
> (e.g. Paninian `k1/k2/vmod` vs UD `nsubj/obj/obl` have zero overlap). Never
> mix sentences from differently annotated corpora in one `generate_variants`
> call.

#### Permutation strategy

| Condition | Strategy |
|---|---|
| N! ≤ 100,000 | Exhaustive enumeration |
| N! > 100,000 | Random sampling (up to 500 valid permutations, 100,000 attempts) |

#### Returns

A single `pandas.DataFrame` (`pairs_df`) with exactly these columns:

| Column | Description |
|---|---|
| `Sent_ID` | Sentence identifier from CoNLL-U metadata |
| `Variant_ID` | `{Sent_ID}_v{n}` |
| `ML_Label` | Flip signal: `1` = reference-first, `0` = flipped (variant-first); balanced 50/50 |
| `Reference_Sentence` | Reference surface text |
| `Variant_Sentence` | Variant surface text |

Feature columns (`Ref_Features`, `Var_Features`, `Delta_DL`, `IS_*`, …) are added
later by scorers — see the scoring section below.

**File saved** (when `output_dir` is given)
- `reference_variant_pairs.csv`

```python
pairs_df = generate_variants(passed, output_dir="output/")
```

---

### `summarize(passed_list, rejected_df)`

Returns a dict of filter statistics and corpus-level numbers. Works with the
output of `filter_sentences` or any manually concatenated rejection DataFrame.

```python
summary = summarize(passed, rejected_df)
```

**Keys**

| Key | Type | Description |
|---|---|---|
| `total_input` | int | Total sentences fed into the first filter |
| `total_passed` | int | Sentences that cleared all filters |
| `total_rejected` | int | Sentences rejected by any filter |
| `pass_rate` | float | `total_passed / total_input` |
| `rejection_counts` | dict | `{reason: count}` for every rejection reason string |
| `filter_order` | list | `[{filter, rejected}, …]` — one entry per filter stage in pipeline order (7 entries total) |
| `avg_sentence_length` | float | Mean token count of passed sentences |
| `avg_phrase_count` | float | Mean preverbal constituent count of passed sentences |
| `phrase_count_distribution` | dict | `{n_phrases: sentence_count}` |

The `filter_order` list groups all `"Bad Root: …"` variants under a single
`"Bad Root"` entry so each stage is represented exactly once regardless of how
many distinct POS tags triggered it.

---

## Corpus compatibility notes

### Paninian-annotated corpora (e.g. ISCNLP, HDTB-Paninian)

Works out of the box with default parameters. The default `allowed_root_pos`
(`VERB / AUX / VM / VAUX`) covers all verbal roots in Paninian annotation.

### UD-annotated corpora (e.g. UD Hindi-HDTB)

Two adjustments are typically needed:

1. **`allowed_root_pos`** — UD marks copular predicates (adjective/noun) as
   the syntactic root. Pass `allowed_root_pos=["VERB", "AUX", "NOUN", "ADJ", "PROPN"]`
   to `filter_sentences` or `filter_bad_root` to recover these sentences.

2. **`filter_punct_constituents`** fires on UD corpora where discourse
   connectors are followed by a comma that is separately attached to the root
   (`"लेकिन,"`, `"बहरहाल,"`, etc.). The filter removes these automatically —
   no configuration needed.

The `check_pos` signal in `filter_negatives` (which looks for a `NEG` POS tag)
is a no-op on UD corpora; all negation is correctly caught by `check_features`
(`Polarity=Neg`). Leave both signals enabled.

---

## Notes

- **Stanza model**: downloaded automatically to `~/stanza_resources/` on first use of `.txt` or raw-string input. Subsequent calls use the cached pipeline.
- **Identical pairs**: a tiny fraction of (reference, variant) pairs may have identical surface text when a sentence contains reduplicated words as separate constituents (e.g. "हंसते हंसते"). This is expected and has no effect on training.
- **Features live in `scoring`**: `generate_variants` emits only surface pairs + `ML_Label`. Run the `dependency_length` / `information_status` scorers (or your own) to add feature columns; their diffs (`Delta_DL`, `Delta_IS`, …) are computed centrally from `ML_Label`.
- **Reproducibility**: set `seed` in `generate_variants()` to control random permutation sampling.
- **Deprel scheme mismatch**: Paninian (`k1`, `k2`, `vmod` …) and UD (`nsubj`, `obj`, `obl` …) deprel labels have zero overlap. Do not mix sentences from corpora with different annotation schemes in a single `generate_variants` call.

---

## Web app

A small FastAPI front end in `webapp/` runs the full pipeline in the browser.
It only *imports* the logic packages — no NLP code lives in `webapp/`.

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` (interactive API docs at `/docs`).

- **Input**: one `.conllu` or `.txt` upload (`.txt` = one Hindi sentence per
  line, parsed with Stanza — the first `.txt` upload downloads the Hindi
  model, so it is slow once).
- **Options**: min preverbal phrases, max variants per sentence, annotation
  scheme preset (Paninian / UD root POS), grammar filter on/off, and which
  pair scorers to run.
- **Grammar filter**: when on (default), only reorderings whose adjacent
  deprel bigrams occur in the uploaded corpus are kept — the library's
  standard behaviour. For small uploads (roughly < 100 sentences) the
  observed bigram set is usually too sparse to license *any* reordering,
  yielding zero variants; uncheck the box to allow all permutations.
- **Downloads per stage** (available as soon as each stage finishes):
  `parsed.conllu` (only for `.txt` input), `passed_sentences.csv`,
  `rejected_sentences.csv`, `summary.json`, and `variants.csv`
  (reference/variant pairs with `Sent_ID`/`Variant_ID` plus scorer columns).
- Jobs run on a 2-worker queue; results expire after one hour.

### Adding a scorer (group members)

Scorers live in the `scoring/` package — you never need to touch the web code.
Create one file, e.g. `scoring/my_scorer.py`:

```python
from .base import Scorer

class MyScorer(Scorer):
    name = "my_scorer"                      # shown as a checkbox in the UI
    description = "One-line description."

    def score(self, pairs_df):
        df = pairs_df.copy()
        df["My_Ref_Score"] = ...            # reference-side value (true role)
        df["My_Var_Score"] = ...            # variant-side value (true role)
        return df

    def deltas(self):
        # Diff is computed centrally from ML_Label at the end of the pipeline.
        return [("Delta_My",
                 lambda row: row["My_Ref_Score"],
                 lambda row: row["My_Var_Score"])]
```

The web app discovers it automatically on restart. `pairs_df` is the
`generate_variants` output (`Sent_ID | Variant_ID | ML_Label |
Reference_Sentence | Variant_Sentence`); add columns, never drop or reorder
rows. Emit raw `*_Reference` / `*_Variant` (true-role) scores and declare the
diff via `deltas()` — don't compute it yourself.

See `scoring/example_scorer.py` for a copy-paste template, and
`scoring/dl_scorer.py` / `scoring/is_scorer.py` for the two built-in scorers.

#### The central diff step (`deltas` + `ML_Label`)

After all selected scorers run, `apply_scorers` computes each scorer's declared
deltas from the `ML_Label` flip signal, so **every feature is differenced with
the same orientation**:

```
Delta_<name> = ref - var   if ML_Label == 1   (reference presented first)
Delta_<name> = var - ref   if ML_Label == 0   (flipped, variant first)
```

This is why scorers store true-role `*_Reference` / `*_Variant` values and only
*declare* their diff: as features are added one by one, their deltas are all
produced consistently in one place.

#### Scorers that need corpus context

`pairs_df` carries only surface strings (the generator computes no features). A
scorer that needs more — the dependency parse, or the **preceding sentence** —
can declare an optional `context` parameter on `score`. `apply_scorers` inspects
the signature and passes `context` only to scorers that ask for it, so
single-argument scorers keep working unchanged:

```python
def score(self, pairs_df, context=None):
    corpus = context["corpus"]          # helpers.CorpusContext
    passed = context["passed"]          # filter output (parse / constituents)
    scheme = context["scheme"]          # "ud" or "paninian"
    ...
```

The pipeline builds this read-only dict per job. Treat it as read-only — for
the web app it is shared across rows and concurrent jobs. `context` is passed
as a call argument (never stored on the shared scorer instance), so it is
thread-safe.

The two recurring building blocks a scorer needs — the **preceding sentence**
and the **variant's reconstructed tree** — live in the `helpers` package (see
below), so you rarely write either from scratch.

## Helpers (`helpers/`)

Shared, scorer-agnostic primitives. Depends only on `conllu`, so any logic
package may import it.

### Preceding-sentence lookup

`helpers.CorpusContext` indexes the full, pre-filter corpus by `sent_id` so any
scorer can fetch the sentences before a given one — the generic solution to the
"context sentence" problem.

```python
from helpers import build_corpus_context, get_previous_sentences

# Repeated lookups — build the index once:
ctx = build_corpus_context(all_sentences)   # all_sentences from load_input()
ctx.previous("doc_s42", n=1)                  # -> List[TokenList] (oldest first)
ctx.previous_text("doc_s42", n=2)             # -> str (concatenated surface)

# One-off lookup — no index to manage:
get_previous_sentences(all_sentences, "doc_s42", n=1)   # -> List[TokenList]
```

(`build_corpus_context` / `CorpusContext` are also re-exported from `scoring`
for backward compatibility.) Build from the **complete** corpus in load order
(before filtering) so a sentence dropped by the filters still counts as the
textual predecessor of the next one. Returns `[]` at the start of the corpus or
for an unknown `sent_id`.

### Variant reconstruction

A variant is stored only as a surface string — there is no CoNLL-U for it.
`helpers.rebuild_variant_tree` rebuilds its reordered, re-indexed dependency
tree from the reference parse + the variant surface string, so any scorer that
measures something positional on a variant (dependency length, …) can reuse it:

```python
from helpers import rebuild_variant_tree

vt = rebuild_variant_tree(reference_sentence, constituents, root_id, variant_sentence)
vt.tokens          # reordered, re-indexed token dicts (the "variant CoNLL-U")
vt.root_id         # new root id after re-indexing
vt.constituents    # variant constituents, re-grouped (lengths preserved)
```

It inherits the relations from the reference parse and only renumbers token IDs
and the head pointers that reference them (the variant is a re-ordering of the
same words). Lower-level pieces are also exported: `recover_permutation`,
`reindex_tokens`, `block_start_index`. Pass `perm=` to skip block-matching when
the constituent order is already known.

### Built-in scorer: `dependency_length`

The dependency-length feature scorer (dependency-length-minimization, after
Gildea & Jaeger 2015). It reconstructs each variant's reordered, re-indexed
dependency tree (from the reference parse in `context["passed"]` plus the
variant surface string) and extracts a 5-element feature vector for both orders.

Feature vector (5 elements, per-arc length = `arc_length − 1`):

| Index | Name | Description |
|---|---|---|
| 0 | `total_DL` | Sentence-level total dependency length |
| 1 | `last_DL` | Dep-head distance of the constituent immediately before the verb |
| 2 | `second_last_DL` | Dep-head distance of the 2nd-closest constituent |
| 3 | `last_len` | Token count of the constituent immediately before the verb |
| 4 | `second_last_len` | Token count of the 2nd-closest constituent |

Columns added to `variants.csv`:

| Column | Description |
|---|---|
| `Ref_Features` | The 5-vector for the reference order (true role) |
| `Var_Features` | The 5-vector for the variant order (true role) |
| `Delta_DL` | Advantage: the `ML_Label`-oriented difference of `total_DL` (computed by the central diff step) |

Needs `context["passed"]` (the filter output). Without it the feature columns
are emitted as zero vectors.

### Built-in scorer: `information_status`

Information Status (IS) / givenness, after Ranjan & van Schijndel (2024). For
each pair it scores whether the **given** phrase (already mentioned in the
previous sentence, or a pronoun) precedes the **new** phrase, comparing the
**subject** and **object** of the clause:

| Order | Score |
|---|---|
| Given → New | `+1` |
| New → Given | `-1` |
| Given → Given / New → New | `0` |

Columns added to `variants.csv`:

| Column | Description |
|---|---|
| `IS_Reference` | IS score of the reference order (true role) |
| `IS_Variant` | IS score of the variant order (true role) |
| `Delta_IS` | Advantage produced by the central diff step: `ML_Label`-oriented difference of the IS scores. On reference-first rows this is `IS_Reference − IS_Variant` (positive = reference adheres to given-before-new, paper footnote 6); on the balanced 50% of flipped rows the sign is reversed. |

A phrase is **GIVEN** if its head is a pronoun, or if any of its content-word
lemmas appear among the previous sentence's content-word lemmas (lemma-vs-lemma
overlap — more robust for inflected Hindi than the original lemma-vs-surface
comparison, so exact scores differ from the reference paper's CSV by design).
Pairs where a subject or object cannot be identified score `0`.

**Annotation scheme** (`context["scheme"]`) selects the relation/POS sets:

| | UD | Paninian |
|---|---|---|
| subject rels | `nsubj`, `csubj` | `k1` |
| object rels | `obj`, `iobj` | `k2`, `k4` |
| pronoun POS | `PRON` | `PRP`, `PR` |
| content POS | `NOUN PROPN VERB ADJ ADV NUM` | `NN NNC NNP NNPC VM JJ RB QC QCC` |

> **Conversion note:** raw-string / `.txt` input is parsed by Stanza, which
> emits **UD** annotation — choose the UD scheme for it. The Paninian scheme is
> meaningful only for pre-annotated Paninian `.conllu` uploads (e.g. the
> bundled ISCNLP corpus). In the web app the scheme is chosen by the
> "Annotation scheme" selector, which also drives root-POS handling.

Library usage:

```python
from stanza_parser import load_input
from filtering import filter_sentences
from variants import generate_variants
from scoring import apply_scorers, build_corpus_context

sentences = load_input("fact_media_news_ISCNLP.conllu")
passed, _, _ = filter_sentences(sentences)
pairs_df = generate_variants(passed)

pairs_df = apply_scorers(
    pairs_df,
    ["dependency_length", "information_status"],
    context={
        "corpus": build_corpus_context(sentences),
        "passed": passed,
        "scheme": "paninian",
    },
)
```
