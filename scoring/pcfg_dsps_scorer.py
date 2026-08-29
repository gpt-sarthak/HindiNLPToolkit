"""
scoring.pcfg_dsps_scorer
========================
Berkeley DS-PS PCFG whole-sentence surprisal scorer.

For each (reference, variant) pair it computes the Berkeley Parser sentence
log-likelihood of each word order under the HDTB DS-PS grammar, negates it into
a surprisal, and declares ``Delta_PCFG``, oriented centrally by ``ML_Label``.

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
Surprisal is exactly ``-log P``, so the parser's log-likelihood is negated on
the way in and the stored value is a **surprisal in nats** (lower = the grammar
finds this order more probable, so easier).  That matches every other surprisal
scorer's direction.

This targets the *same quantity* as the ``surprisal`` scorer — the total
surprisal of the sentence — but arrives at it differently: a different grammar
(HUTB DS-PS vs an HDTB-derived model), a different engine (the Berkeley Java
parser vs the Taru synproc C++ left-corner parser), nats vs bits, and a single
whole-sentence figure rather than a sum of per-word increments.

Discovered automatically by the scoring package — appears as the
``berkeley_pcfg`` checkbox in the UI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

from helpers import heartbeat, register_process

from .base import Scorer

_PKG_DIR = Path(__file__).resolve().parent
_JAR = _PKG_DIR.parent / "taru" / "external_resources" / "berkeleyparser" / "berkeleyParser.jar"
_GRAMMAR = _PKG_DIR / "models" / "hdtb_dsps_grammar"


def _available() -> bool:
    return _JAR.is_file() and _GRAMMAR.is_file()


def _score_pcfg_live(sentences: List[str]) -> Dict[str, float]:
    """Run the Berkeley Parser on a batch of sentences, returning
    ``{sentence: surprisal_in_nats}`` (the negated sentence log-likelihood).
    Sentences that fail to parse (or any failure to invoke the parser) are
    simply absent from the result."""
    if not _available() or not sentences:
        return {}

    inp = "\n".join(sentences) + "\n"
    try:
        # Popen rather than subprocess.run so the JVM can be registered with the
        # job's cancel token: this is one blocking call for the whole batch, so
        # a cooperative heartbeat cannot interrupt it — the killer must be able
        # to terminate the process itself.  Outside a job the registration is a
        # no-op and this behaves exactly like the old subprocess.run.
        with subprocess.Popen(
            ["java", "-Xmx4g", "-jar", str(_JAR),
             "-gr", str(_GRAMMAR), "-sentence_likelihood"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        ) as proc, register_process(proc):
            try:
                stdout, _stderr = proc.communicate(input=inp, timeout=300)
            except Exception:
                proc.kill()          # don't leave the JVM behind on a timeout
                raise
    except Exception:
        return {}

    scores: Dict[str, float] = {}
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    # The parser emits one likelihood line per input sentence, in order.
    for sent, line in zip(sentences, lines):
        try:
            log_prob = float(line.split("\t")[0])
            # Guard the RAW log-likelihood: the parser returns -inf for
            # unparseable sentences, and those must be dropped *before* the
            # negation below (negating -inf would yield +inf, i.e. a sentence
            # that looks maximally surprising rather than unscored).
            if log_prob > -1e10:
                scores[sent] = -log_prob  # surprisal = -log P
        except (ValueError, IndexError):
            pass
    return scores


class PCFGDSPSScorer(Scorer):
    name = "berkeley_pcfg"
    display_name = "DSPS PCFG Surprisal"
    description = (
        "How surprising each word order's full constituency structure is, "
        "scored whole-sentence rather than word-by-word.\n"
        "Feature in CSV: Delta_PCFG."
    )
    trained_on = "HUTB - 13,282 DS-PS constituency trees"
    built_with = "Berkeley Parser, DS-PS grammar"
    notes = "surprisal in nats — lower = easier to process"
    higher_is_more_natural = {"Delta_PCFG": False}  # lower surprisal = more natural

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
        # A cancel kills the JVM mid-batch, leaving partial output; unwind here
        # rather than publishing a half-scored column.
        heartbeat()

        df["PCFG_Reference"] = [scores.get(str(s), float("nan")) for s in df["Reference_Sentence"]]
        df["PCFG_Variant"] = [scores.get(str(s), float("nan")) for s in df["Variant_Sentence"]]
        return df

    def deltas(self):
        return [("Delta_PCFG",
                 lambda row: row["PCFG_Reference"],
                 lambda row: row["PCFG_Variant"])]
