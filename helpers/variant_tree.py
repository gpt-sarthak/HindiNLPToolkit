"""
helpers.variant_tree
====================
Rebuild the dependency tree of a *variant* sentence.

A variant produced by ``variants.generate_variants`` is stored only as a surface
string — there is no CoNLL-U for it. Any scorer that needs the variant's
*structure* (token positions, head pointers, per-constituent grouping) must
reconstruct it from the reference parse plus the variant surface string. That
reconstruction is generic — not specific to dependency length — so it lives here
for every scorer to reuse.

    from helpers import rebuild_variant_tree
    vt = rebuild_variant_tree(reference_sentence, constituents, root_id,
                              variant_sentence)
    vt.tokens          # reordered, re-indexed token dicts (the "variant CoNLL-U")
    vt.root_id         # new root id after re-indexing
    vt.constituents    # variant constituents, re-grouped (lengths preserved)

The relations are inherited from the reference parse; only positions (token ids)
and the head pointers that reference them are renumbered, because the variant is
a re-ordering of the same words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def block_start_index(haystack: List[str], needle: List[str]) -> int:
    """Index of the first contiguous occurrence of *needle* in *haystack*,
    or -1 if absent.  Used to locate a constituent's token block inside a
    variant's surface tokens."""
    if not needle:
        return -1
    last = len(haystack) - len(needle)
    for i in range(last + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def recover_permutation(
    variant_forms: List[str], constituents: List[List[dict]]
) -> List[int]:
    """Recover the constituent order of a variant by locating each reference
    constituent's token block in the variant surface forms.  Returns a list of
    original constituent indices in the order they appear in the variant.
    Falls back to the identity order if any block cannot be located."""
    starts = []
    for idx, const in enumerate(constituents):
        block = [tok["form"] for tok in const]
        start = block_start_index(variant_forms, block)
        if start < 0:
            return list(range(len(constituents)))  # cannot disambiguate
        starts.append((start, idx))
    starts.sort()
    return [idx for _, idx in starts]


def reindex_tokens(ordered_tokens: List[dict]) -> Tuple[List[dict], Optional[int]]:
    """Re-number token IDs sequentially (1, 2, 3 …) after reordering and update
    all head references to the new IDs.  Heads that no longer map to a valid
    token become 0.  Returns ``(new_token_list, new_root_id)``."""
    old_to_new = {tok["id"]: i + 1 for i, tok in enumerate(ordered_tokens)}
    new_sentence = []
    new_root_id = None
    for i, tok in enumerate(ordered_tokens):
        new_tok = dict(tok)
        new_tok["id"] = i + 1
        if tok["head"] == 0:
            new_tok["head"] = 0
            new_root_id = new_tok["id"]
        elif tok["head"] in old_to_new:
            new_tok["head"] = old_to_new[tok["head"]]
        else:
            new_tok["head"] = 0
        new_sentence.append(new_tok)
    return new_sentence, new_root_id


@dataclass
class VariantTree:
    """A reconstructed variant: the reordered, re-indexed tokens (the in-memory
    "variant CoNLL-U"), its new root id, and its constituents re-grouped to match
    the new token order."""
    tokens: List[dict] = field(default_factory=list)
    root_id: Optional[int] = None
    constituents: List[List[dict]] = field(default_factory=list)


def rebuild_variant_tree(
    reference_sentence,
    constituents: List[List[dict]],
    root_id: int,
    variant_sentence: str,
    perm: Optional[List[int]] = None,
) -> VariantTree:
    """
    Rebuild a variant's dependency tree from the reference parse and the variant
    surface string.

    Parameters
    ----------
    reference_sentence : iterable of token dicts — the reference parse.
    constituents       : the reference preverbal constituents (subtrees).
    root_id            : reference root token id.
    variant_sentence   : the variant surface string.
    perm               : optional constituent order (list of original indices).
                         Recovered from *variant_sentence* if not given — pass it
                         to skip the block-matching when the order is known.

    Returns
    -------
    VariantTree — empty (``tokens=[]``) if the sentence has no preverbal block.
    """
    flat_preverbal_ids = {tok["id"] for const in constituents for tok in const}
    if not flat_preverbal_ids:
        return VariantTree()
    start_pos = min(flat_preverbal_ids)
    prefix = [t for t in reference_sentence
              if t["id"] < start_pos and t["id"] not in flat_preverbal_ids]
    suffix = [t for t in reference_sentence
              if t["id"] > start_pos and t["id"] not in flat_preverbal_ids]

    if perm is None:
        perm = recover_permutation(variant_sentence.split(" "), constituents)
    permuted_consts = [constituents[i] for i in perm]
    new_order = prefix + [tok for const in permuted_consts for tok in const] + suffix
    var_tokens, var_root = reindex_tokens(new_order)

    # Re-group variant constituents by cursor — lengths are preserved.
    var_consts: List[List[dict]] = []
    curr = len(prefix)
    for old_const in permuted_consts:
        length = len(old_const)
        var_consts.append(var_tokens[curr: curr + length])
        curr += length

    return VariantTree(tokens=var_tokens, root_id=var_root, constituents=var_consts)
