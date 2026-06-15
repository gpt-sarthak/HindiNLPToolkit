"""
helpers.corpus_context
=======================
Look up the sentence(s) that precede a given sentence in a corpus.

A scorer (or any analysis) often needs the *previous* sentence — e.g. for
givenness / Information Status. ``pairs_df`` only identifies a row's source
sentence by ``sent_id`` and says nothing about what came before it, so this
helper indexes the full, pre-filter corpus by ``sent_id``.

Two ways to use it:

    # Repeated lookups — build the index once:
    from helpers import build_corpus_context
    ctx = build_corpus_context(all_sentences)     # all_sentences from load_input()
    ctx.previous("doc_s42", n=1)                   # -> List[TokenList]
    ctx.previous_text("doc_s42", n=2)              # -> str

    # One-off lookup — no index to manage:
    from helpers import get_previous_sentences, get_previous_text
    get_previous_sentences(all_sentences, "doc_s42", n=1)
    get_previous_text(all_sentences, "doc_s42", n=1)

Pass the *complete* corpus in load order (before filtering) so that a sentence
dropped by the filters still counts as the textual predecessor of the next one.
"""

from __future__ import annotations

from typing import Dict, List

from conllu import TokenList


def _sent_id(sentence: TokenList, fallback: str) -> str:
    """Return a sentence's CoNLL-U ``sent_id`` metadata, or *fallback*."""
    return sentence.metadata.get("sent_id", fallback)


def _surface(sentence: TokenList) -> str:
    """Space-joined surface form of a sentence (integer-id tokens only)."""
    return " ".join(
        tok["form"] for tok in sentence if isinstance(tok["id"], int)
    )


class CorpusContext:
    """
    Index of a corpus in load order, queryable by ``sent_id``.

    Parameters
    ----------
    sentences : List[TokenList]
        The full corpus in load order (the output of ``load_input``), ideally
        *before* filtering so that the textual predecessor of every sentence is
        preserved even if it was later filtered out.

    Notes
    -----
    If two sentences share a ``sent_id`` the first occurrence wins for lookups;
    the ordered backing list still holds every sentence, so ``previous`` always
    reflects true corpus order.
    """

    def __init__(self, sentences: List[TokenList]) -> None:
        self._sentences: List[TokenList] = list(sentences)
        self._position: Dict[str, int] = {}
        for idx, sent in enumerate(self._sentences):
            sid = _sent_id(sent, f"index_{idx}")
            # First occurrence wins — keeps lookups stable for duplicate ids.
            self._position.setdefault(sid, idx)

    def previous(self, sent_id: str, n: int = 1) -> List[TokenList]:
        """
        Return up to *n* sentences immediately preceding *sent_id* in corpus
        order, oldest first.  Empty list if *sent_id* is unknown or is at the
        start of the corpus.
        """
        if n < 1:
            return []
        pos = self._position.get(sent_id)
        if pos is None:
            return []
        start = max(0, pos - n)
        return self._sentences[start:pos]

    def previous_text(self, sent_id: str, n: int = 1) -> str:
        """Surface text of the previous *n* sentences, space-joined."""
        return " ".join(_surface(sent) for sent in self.previous(sent_id, n))

    def __len__(self) -> int:
        return len(self._sentences)


def build_corpus_context(sentences: List[TokenList]) -> CorpusContext:
    """Construct a :class:`CorpusContext` from a loaded sentence list."""
    return CorpusContext(sentences)


def get_previous_sentences(
    sentences: List[TokenList], sent_id: str, n: int = 1
) -> List[TokenList]:
    """
    One-shot convenience: the up-to-*n* sentences preceding *sent_id* in
    *sentences* (load order), oldest first.

    Builds a :class:`CorpusContext` internally — for many lookups, build one
    ``CorpusContext`` yourself and reuse it instead.
    """
    return CorpusContext(sentences).previous(sent_id, n)


def get_previous_text(
    sentences: List[TokenList], sent_id: str, n: int = 1
) -> str:
    """One-shot convenience: surface text of the previous *n* sentences."""
    return CorpusContext(sentences).previous_text(sent_id, n)
