"""
stanza_parser
=============
Loads Hindi sentences from any source into conllu TokenList objects.

    from stanza_parser import load_input             # unified loader
    from stanza_parser import parse_text, load_txt   # lower-level helpers
"""

from .parser import load_input, load_txt, parse_text

__all__ = ["load_input", "parse_text", "load_txt"]
