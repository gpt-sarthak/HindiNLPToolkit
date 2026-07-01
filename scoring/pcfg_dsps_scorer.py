"""
scoring.pcfg_dsps_scorer
========================
Berkeley DS-PS PCFG sentence log-likelihood scorer.

For each (reference, variant) pair it computes the Berkeley Parser sentence
log-likelihood of each word order under the HDTB DS-PS grammar and declares
``Delta_PCFG``, oriented centrally by ``ML_Label``.

Scoring runs the Berkeley Parser as one batched JVM call per ``score`` over the
unique surfaces in the table::

    java -Xmx4g -jar berkeleyParser.jar -gr <grammar> -sentence_likelihood

The jar is the one already bundled for the Taru tool
(``taru/external_resources/berkeleyparser/berkeleyParser.jar``); only the
1.7 MB DS-PS grammar (``scoring/models/hdtb_dsps_grammar``) is specific to this
scorer.  Java must be on PATH (it is, in the Docker image).  If the jar/grammar
is missing, Java is unavailable, or a sentence is unparseable, that score is
``NaN``.

Units / direction
-----------------
The value is a **log-likelihood** (higher = more probable), NOT a surprisal.
This scorer is distinct from the ``surprisal`` scorer, which is *incremental*
constituency surprisal from the Taru synproc engine on a different (hdtb_fresh)
grammar.

Discovered automatically by the scoring package — appears as the
``berkeley_pcfg`` checkbox in the UI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

from .base import Scorer

_PKG_DIR = Path(__file__).resolve().parent
_JAR = _PKG_DIR.parent / "taru" / "external_resources" / "berkeleyparser" / "berkeleyParser.jar"
_GRAMMAR = _PKG_DIR / "models" / "hdtb_dsps_grammar"


def _available() -> bool:
    return _JAR.is_file() and _GRAMMAR.is_file()


def _score_pcfg_live(sentences: List[str]) -> Dict[str, float]:
    """Run the Berkeley Parser on a batch of sentences, returning
    ``{sentence: log_likelihood}``.  Sentences that fail to parse (or any
    failure to invoke the parser) are simply absent from the result."""
    if not _available() or not sentences:
        return {}

    inp = "\n".join(sentences) + "\n"
    try:
        res = subprocess.run(
            ["java", "-Xmx4g", "-jar", str(_JAR),
             "-gr", str(_GRAMMAR), "-sentence_likelihood"],
            input=inp,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
    except Exception:
        return {}

    scores: Dict[str, float] = {}
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    # The parser emits one likelihood line per input sentence, in order.
    for sent, line in zip(sentences, lines):
        try:
            log_prob = float(line.split("\t")[0])
            if log_prob > -1e10:  # parser returns -inf for unparseable sentences
                scores[sent] = log_prob
        except (ValueError, IndexError):
            pass
    return scores


class PCFGDSPSScorer(Scorer):
    name = "berkeley_pcfg"
    description = (
        "Berkeley DS-PS PCFG sentence log-likelihood of each word order. "
        "Advantage: Delta_PCFG."
    )
    trained_on = "HUTB - 13,282 DS-PS constituency trees"
    built_with = "Berkeley Parser (PCFGLA) grammar, -sentence_likelihood"
    notes = "log-likelihood — higher = more probable"

    def score(self, pairs_df):
        df = pairs_df.copy()
        if df.empty:
            df["PCFG_Reference"] = []
            df["PCFG_Variant"] = []
            return df

        surfaces = list(dict.fromkeys(
            [str(s) for s in df["Reference_Sentence"]]
            + [str(s) for s in df["Variant_Sentence"]]
        ))
        scores = _score_pcfg_live(surfaces)

        df["PCFG_Reference"] = [scores.get(str(s), float("nan")) for s in df["Reference_Sentence"]]
        df["PCFG_Variant"] = [scores.get(str(s), float("nan")) for s in df["Variant_Sentence"]]
        return df

    def deltas(self):
        return [("Delta_PCFG",
                 lambda row: row["PCFG_Reference"],
                 lambda row: row["PCFG_Variant"])]
