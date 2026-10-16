"""
scoring._lstm_engine
====================
Batched, prefix-sharing sentence-surprisal engine shared by all four LSTM
scorers (``lstm``, ``adaptive_lstm``, ``lstm_wiki50m``,
``adaptive_lstm_wiki50m``).

Private helper: it defines no ``Scorer`` subclass, so plugin discovery skips it.
It knows nothing about tokenisation — each scorer encodes its own sentences
(the two model generations frame them differently) and hands over token ids.

How a call is scored
--------------------
1. Identical sequences are scored once.
2. The unique sequences are sorted (by length, then token ids) so sequences
   that share a prefix sit next to each other.  Results are written back to the
   caller's positions, so the caller's order never matters.
3. The LSTM runs over the sequences in chunks of ``batch_size``.  Within one
   reference group every variant has the same length, so a chunk needs no
   padding at all.
4. **Prefix cache.**  The hidden state after a token prefix depends only on
   that prefix, so two sequences that start the same way predict their next
   word from the *same* distribution.  Each unique prefix gets one node, and
   the vocabulary projection + ``log_softmax`` — by far the most expensive step
   (a 30k-word output at every position) — runs once per node instead of once
   per position.  Same arithmetic, fewer rows.  With the cache off, every
   position is its own node and the engine is a plain batched scorer.

Totals are summed in float64, matching the older scorers, which accumulated
Python floats one ``.item()`` at a time.

Settings (environment, read per call so a demo box can be tuned without a
rebuild):

``HNT_LSTM_BATCH``         sequences per LSTM chunk (default 16 — measured best
                           on a 4-thread CPU; 128 was slower)
``HNT_LSTM_PREFIX_CACHE``  ``1`` / ``0`` to switch the prefix cache on / off
``HNT_TORCH_THREADS``      torch intra-op threads, applied once at model load
                           (unset = torch's default)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_DEFAULT_BATCH = 16
_DEFAULT_PREFIX_CACHE = True
# Rows of the vocabulary projection computed at once: bounds the size of the
# (rows x vocab) logits block, ~60 MB at a 30k vocabulary.
_PROJECTION_ROWS = 512

_threads_configured = False


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
        return value if value > 0 else default
    except ValueError:
        return default


def batch_size_setting() -> int:
    return _env_int("HNT_LSTM_BATCH", _DEFAULT_BATCH)


def prefix_cache_setting() -> bool:
    raw = os.environ.get("HNT_LSTM_PREFIX_CACHE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return _DEFAULT_PREFIX_CACHE


def configure_torch_threads() -> None:
    """Apply ``HNT_TORCH_THREADS`` once per process (called by the model
    loaders).  Unset or invalid leaves torch's own default alone."""
    global _threads_configured
    if _threads_configured:
        return
    _threads_configured = True
    threads = _env_int("HNT_TORCH_THREADS", 0)
    if threads:
        import torch

        torch.set_num_threads(threads)


def sequence_surprisals(
    sequences: Sequence[Sequence[int]],
    model,
    device,
    *,
    pad_id: int = 0,
    batch_size: Optional[int] = None,
    prefix_cache: Optional[bool] = None,
    on_chunk: Optional[Callable[[], None]] = None,
) -> List[float]:
    """
    Total next-token surprisal (nats) of each id sequence, in the order given.

    Every token after the first is predicted from its left context; a sequence
    shorter than two ids scores ``0.0``.  *model* must expose ``embedding``,
    ``lstm`` and ``fc`` (the shared ``LSTMLanguageModel``) and is used
    read-only, under ``torch.no_grad``.

    ``on_chunk`` is called once per LSTM chunk and once per projection chunk —
    the scorers pass ``helpers.heartbeat`` so a long run stays cancellable.
    """
    import torch

    if batch_size is None:
        batch_size = batch_size_setting()
    if prefix_cache is None:
        prefix_cache = prefix_cache_setting()

    out = [0.0] * len(sequences)

    # 1. De-duplicate; short sequences stay 0.0.
    where: Dict[Tuple[int, ...], List[int]] = {}
    for i, seq in enumerate(sequences):
        key = tuple(int(t) for t in seq)
        if len(key) >= 2:
            where.setdefault(key, []).append(i)
    if not where:
        return out

    # 2. Sort so shared prefixes are adjacent (and lengths are uniform per chunk).
    uniq = sorted(where, key=lambda s: (len(s), s))

    # 3. Assign a node to every (sequence, input position).  A node stands for
    #    the prefix seq[:d+1]; its hidden state predicts seq[d+1].  With the
    #    cache on, nodes are shared through a trie keyed (parent, token).
    trie: Dict[Tuple[int, int], int] = {}
    node_first: List[Tuple[int, int]] = []      # node -> (sequence, position)
    seq_nodes: List[List[int]] = []
    for si, seq in enumerate(uniq):
        nodes = []
        parent = -1
        for d in range(len(seq) - 1):
            if prefix_cache:
                key = (parent, seq[d])
                node = trie.get(key)
                if node is None:
                    node = len(node_first)
                    trie[key] = node
                    node_first.append((si, d))
                parent = node
            else:
                node = len(node_first)
                node_first.append((si, d))
            nodes.append(node)
        seq_nodes.append(nodes)

    # Which (node, next token) log-probs are needed — a shared prefix can be
    # followed by different tokens in different sequences.
    pair_id: Dict[Tuple[int, int], int] = {}
    seq_pairs: List[List[int]] = []
    for si, seq in enumerate(uniq):
        ids = []
        for d, node in enumerate(seq_nodes[si]):
            key = (node, seq[d + 1])
            pid = pair_id.get(key)
            if pid is None:
                pid = len(pair_id)
                pair_id[key] = pid
            ids.append(pid)
        seq_pairs.append(ids)

    # Nodes whose hidden state must be kept, grouped by the sequence that
    # first produced them.
    keep_by_seq: Dict[int, List[Tuple[int, int]]] = {}
    for node, (si, d) in enumerate(node_first):
        keep_by_seq.setdefault(si, []).append((node, d))

    hidden_size = model.fc.in_features
    with torch.no_grad():
        # 4. LSTM over the sequences, chunked; keep only the node states.
        node_h = torch.empty((len(node_first), hidden_size), device=device)
        for start in range(0, len(uniq), batch_size):
            if on_chunk is not None:
                on_chunk()
            chunk = uniq[start:start + batch_size]
            width = max(len(s) for s in chunk) - 1
            inp = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            for r, seq in enumerate(chunk):
                inp[r, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            h, _ = model.lstm(model.embedding(inp.to(device)))
            rows, cols, nodes = [], [], []
            for r in range(len(chunk)):
                for node, d in keep_by_seq.get(start + r, ()):
                    rows.append(r)
                    cols.append(d)
                    nodes.append(node)
            if nodes:
                node_h[torch.tensor(nodes, device=device)] = h[
                    torch.tensor(rows, device=device),
                    torch.tensor(cols, device=device)]

        # 5. Project once per node and pick the needed next-token log-probs.
        pair_node = torch.empty(len(pair_id), dtype=torch.long)
        pair_tok = torch.empty(len(pair_id), dtype=torch.long)
        for (node, tok), pid in pair_id.items():
            pair_node[pid] = node
            pair_tok[pid] = tok
        order = torch.argsort(pair_node)          # pairs grouped by node
        sorted_nodes = pair_node[order]
        pair_logp = torch.empty(len(pair_id), dtype=torch.float64)
        for start in range(0, len(node_first), _PROJECTION_ROWS):
            if on_chunk is not None:
                on_chunk()
            stop = min(start + _PROJECTION_ROWS, len(node_first))
            logp = torch.log_softmax(model.fc(node_h[start:stop]), dim=-1).cpu()
            lo = int(torch.searchsorted(sorted_nodes, start))
            hi = int(torch.searchsorted(sorted_nodes, stop))
            sel = order[lo:hi]
            pair_logp[sel] = logp[pair_node[sel] - start, pair_tok[sel]].double()

    # 6. Sum each sequence's log-probs and broadcast to its duplicates.
    for si, seq in enumerate(uniq):
        total = -float(pair_logp[torch.tensor(seq_pairs[si])].sum())
        for i in where[seq]:
            out[i] = total
    return out


# --- pairs-table drivers --------------------------------------------------------

def _group_rows(pairs_df, positions):
    """The reference text and the variant texts of one reference group."""
    ref = str(pairs_df["Reference_Sentence"].iat[positions[0]])
    return ref, [str(pairs_df["Variant_Sentence"].iat[p]) for p in positions]


def score_pairs(pairs_df, model, device, encode: Callable[[str], List[int]],
                *, pad_id: int = 0,
                on_chunk: Optional[Callable[[], None]] = None
                ) -> Tuple[List[float], List[float]]:
    """
    ``(reference_scores, variant_scores)`` for every row, under one fixed model.

    Rows are grouped by reference (``helpers.reference_groups``); each group's
    reference + variants are scored together, so the reference is computed once
    and shares its prefixes with the variants.  Small groups are packed into
    one engine call (up to ``4 x HNT_LSTM_BATCH`` sentences) so a corpus of
    short sentences still fills its batches.  A surface already scored earlier
    in the table is not scored again.
    """
    from helpers import reference_groups

    n = len(pairs_df)
    ref_vals = [float("nan")] * n
    var_vals = [float("nan")] * n
    if n == 0:
        return ref_vals, var_vals

    flush_at = 4 * batch_size_setting()
    known: Dict[str, float] = {}
    pending: List[str] = []                 # surfaces waiting for the next call
    queued: set = set()
    groups = reference_groups(pairs_df)

    def flush():
        if not pending:
            return
        scores = sequence_surprisals([encode(s) for s in pending], model, device,
                                     pad_id=pad_id, on_chunk=on_chunk)
        known.update(zip(pending, scores))
        pending.clear()
        queued.clear()

    for _key, positions in groups:
        ref, variants = _group_rows(pairs_df, positions)
        for text in [ref] + variants:
            if text not in known and text not in queued:
                pending.append(text)
                queued.add(text)
        if len(pending) >= flush_at:
            flush()
    flush()

    for _key, positions in groups:
        ref, variants = _group_rows(pairs_df, positions)
        for p, v in zip(positions, variants):
            ref_vals[p] = known[ref]
            var_vals[p] = known[v]
    return ref_vals, var_vals


def score_group(pairs_df, positions, model, device,
                encode: Callable[[str], List[int]], *, pad_id: int = 0,
                on_chunk: Optional[Callable[[], None]] = None
                ) -> Tuple[float, List[float]]:
    """One reference group under one model (the adaptive scorers' unit, since
    each group has its own adapted weights): ``(reference_score,
    [variant_score per position])``.  Reference and variants share a call."""
    ref, variants = _group_rows(pairs_df, positions)
    scores = sequence_surprisals([encode(s) for s in [ref] + variants], model,
                                 device, pad_id=pad_id, on_chunk=on_chunk)
    return scores[0], scores[1:]


def sgd_step(model, ids: Sequence[int], device, lr: float, *,
             deterministic: bool = True, clip: Optional[float] = None) -> None:
    """
    One plain-SGD step of next-token cross-entropy on the id sequence *ids*,
    applied to *model* in place — the adaptation step of both adaptive scorers.

    Equivalent to ``torch.optim.SGD(model.parameters(), lr).step()`` after
    ``loss.backward()``, but the embedding is updated **sparsely**: its gradient
    is non-zero only on the rows of tokens in *ids* (~20 of 30k), so only those
    rows are written instead of a dense 30k x 256 gradient being built and
    applied.  The touched rows are recorded on the model for
    :class:`AdaptationCopy` to reset.

    ``deterministic`` zeroes the LSTM dropout for the step (see
    ``_lstm_common.adapt_one_step``); ``clip`` optionally clips the global grad
    norm, computed over the same gradients the dense step would see.
    """
    import torch
    import torch.nn.functional as F

    inp = torch.tensor(list(ids[:-1]), dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.tensor(list(ids[1:]), dtype=torch.long, device=device)
    dense_params = [p for n, p in model.named_parameters() if not n.startswith("embedding.")]

    saved_dropout = model.lstm.dropout
    if deterministic:
        model.lstm.dropout = 0.0
    model.train()
    try:
        emb = model.embedding(inp)                      # embedding rows of *inp*
        logits, _ = model.lstm(emb)
        logits = model.fc(logits)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        grads = torch.autograd.grad(loss, [emb] + dense_params)
    finally:
        model.lstm.dropout = saved_dropout
        model.eval()

    emb_grad, dense_grads = grads[0][0], grads[1:]      # (T, E) per input position
    if clip:
        # Same norm as the dense gradient: rows of a repeated token add up.
        rows, inverse = torch.unique(inp[0], return_inverse=True)
        summed = torch.zeros((len(rows), emb_grad.size(1)), device=emb_grad.device)
        summed.index_add_(0, inverse, emb_grad)
        total = torch.sqrt(sum((g.double() ** 2).sum() for g in dense_grads)
                           + (summed.double() ** 2).sum())
        scale = min(1.0, clip / (float(total) + 1e-6))
        emb_grad = emb_grad * scale
        dense_grads = [g * scale for g in dense_grads]
    with torch.no_grad():
        for p, g in zip(dense_params, dense_grads):
            p.add_(g, alpha=-lr)
        model.embedding.weight.index_add_(0, inp[0], emb_grad, alpha=-lr)
    touched = getattr(model, "_hnt_touched_rows", None)
    rows = inp[0].unique()
    model._hnt_touched_rows = rows if touched is None else torch.cat([touched, rows]).unique()


class AdaptationCopy:
    """
    A reusable working copy of a shared base model for per-group adaptation.

    The first :meth:`fresh` deep-copies the base.  Later calls reset that same
    copy in place instead of allocating a new ~65 MB model for every reference,
    and reset only what an adaptation step can have changed: the dense layers
    (LSTM + output projection) and the embedding rows :func:`sgd_step` recorded
    as touched — so adapt the copy **only** through :func:`sgd_step` (both
    ``adapt_one_step`` functions do).  One instance per ``score()`` call, so
    concurrent jobs never share a copy, and the base itself is never written
    to.
    """

    def __init__(self, base_model) -> None:
        self._base = base_model
        self._work = None

    def fresh(self):
        import torch

        if self._work is None:
            import copy

            self._work = copy.deepcopy(self._base)
            self._work._hnt_touched_rows = None
        else:
            base = dict(self._base.named_parameters())
            rows = getattr(self._work, "_hnt_touched_rows", None)
            with torch.no_grad():
                for name, p in self._work.named_parameters():
                    if name == "embedding.weight" and rows is not None:
                        if len(rows):
                            p[rows] = base[name][rows]
                    else:
                        p.copy_(base[name])
            self._work._hnt_touched_rows = torch.empty(0, dtype=torch.long)
        self._work.eval()
        return self._work
