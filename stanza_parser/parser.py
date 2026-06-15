"""
stanza_parser.parser
====================
Converts raw Hindi text or plain-text files to conllu TokenList objects
using the Stanza Hindi NLP pipeline.

The Stanza model is downloaded automatically on first use and cached
in-process so repeated calls do not reload it.

Public functions
----------------
load_input(source)     -> List[TokenList]   unified loader (.conllu / .txt / raw string)
parse_text(text)       -> List[TokenList]   parse a raw Hindi sentence string
load_txt(filepath)     -> List[TokenList]   parse a .txt file (one sentence per line)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from conllu import parse, TokenList

# Module-level cache — pipeline is loaded once per Python session.
_stanza_pipeline_cache: Dict[str, Any] = {}


def _get_stanza_pipeline(lang: str = "hi") -> Any:
    """
    Return a cached Stanza pipeline for *lang*, downloading the model on first
    use.  Raises ImportError if the stanza package is not installed.
    """
    if lang not in _stanza_pipeline_cache:
        try:
            import stanza  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "stanza is required to parse raw text or .txt files.\n"
                "Install it with:  pip install stanza"
            ) from exc
        stanza.download(lang, verbose=False)
        _stanza_pipeline_cache[lang] = stanza.Pipeline(
            lang,
            processors="tokenize,pos,lemma,depparse",
            verbose=False,
        )
    return _stanza_pipeline_cache[lang]


def _stanza_doc_to_conllu_tokens(doc: Any) -> List[TokenList]:
    """
    Serialise a Stanza Doc to a CoNLL-U string and re-parse it with the
    conllu library so downstream code always works with TokenList objects.
    """
    lines: List[str] = []
    for sent_idx, sentence in enumerate(doc.sentences):
        lines.append(f"# sent_id = stanza_{sent_idx + 1}")
        lines.append(f"# text = {sentence.text}")
        for word in sentence.words:
            feats = word.feats if word.feats else "_"
            lines.append(
                f"{word.id}\t{word.text}\t{word.lemma or '_'}\t"
                f"{word.upos or '_'}\t{word.xpos or '_'}\t{feats}\t"
                f"{word.head}\t{word.deprel or '_'}\t_\t_"
            )
        lines.append("")
    return parse("\n".join(lines))


def load_input(source: str) -> List[TokenList]:
    """
    Load Hindi sentences from a file or a raw string.

    Routing logic
    -------------
    Path ending in ``.conllu``  →  parsed with the ``conllu`` library directly.
    Path ending in ``.txt``     →  each non-empty line parsed via Stanza (Hindi).
    Any other string            →  treated as a raw Hindi sentence; parsed via Stanza.

    Parameters
    ----------
    source : str
        File path (ending in ``.conllu`` or ``.txt``) or a Hindi sentence string.

    Returns
    -------
    List[TokenList]

    Raises
    ------
    FileNotFoundError  — path given but file does not exist.
    ValueError         — empty source, or file contains no parseable content.
    ImportError        — Stanza is needed but not installed.

    Examples
    --------
    >>> from stanza_parser import load_input
    >>> sents = load_input("corpus.conllu")
    >>> sents = load_input("sentences.txt")
    >>> sents = load_input("राम घर जाता है।")
    """
    if not source or not source.strip():
        raise ValueError("source must be a non-empty file path or sentence string.")

    path = Path(source)

    if path.suffix.lower() in (".conllu", ".txt"):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if path.suffix.lower() == ".conllu":
            content = path.read_text(encoding="utf-8")
            tokens = parse(content)
            if not tokens:
                raise ValueError(f"No sentences parsed from {source}")
            return tokens

        return load_txt(source)

    return parse_text(source)


def parse_text(text: str) -> List[TokenList]:
    """
    Parse a raw Hindi sentence (or short paragraph) using Stanza.

    Parameters
    ----------
    text : str
        One or more Hindi sentences as a plain string.

    Returns
    -------
    List[TokenList]
        One TokenList per sentence detected by Stanza's tokeniser.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    ImportError
        If stanza is not installed.

    Example
    -------
    >>> from stanza_parser import parse_text
    >>> tokens = parse_text("राम घर जाता है।")
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty Hindi string.")
    nlp = _get_stanza_pipeline("hi")
    doc = nlp(text.strip())
    return _stanza_doc_to_conllu_tokens(doc)


def load_txt(filepath: str) -> List[TokenList]:
    """
    Parse a plain-text file where every non-empty line is one Hindi sentence.

    Each line is run through Stanza independently so that sent_ids are stable
    and reflect the original line number (``txt_line<N>_s<M>``).

    Parameters
    ----------
    filepath : str
        Path to a UTF-8 encoded .txt file.

    Returns
    -------
    List[TokenList]
        One TokenList per sentence (a line may produce multiple TokenLists if
        Stanza's tokeniser splits it further).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file contains no non-empty lines.
    ImportError
        If stanza is not installed.

    Example
    -------
    >>> from stanza_parser import load_txt
    >>> tokens = load_txt("sentences.txt")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No sentences found in {filepath}")
    nlp = _get_stanza_pipeline("hi")
    all_tokens: List[TokenList] = []
    for line_idx, line in enumerate(lines):
        doc = nlp(line)
        parsed = _stanza_doc_to_conllu_tokens(doc)
        for sent_idx, sent in enumerate(parsed):
            sent.metadata["sent_id"] = f"txt_line{line_idx + 1}_s{sent_idx + 1}"
        all_tokens.extend(parsed)
    return all_tokens
