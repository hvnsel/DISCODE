"""
discover_edit_policy.py
=======================

PROTOTYPE ADD-ON.  The actor-critic behind the move-based search in
:mod:`discover_edit_env`: it reads the current equation and scores every move.

Architecture
------------
Each term's tokens are encoded by a small transformer and mean-pooled into one
vector per term.  A second transformer runs over ``[global] + terms`` with NO
positional encoding, so it is permutation-equivariant over term slots: a move
that targets term ``j`` (WRAP, MUL, DELETE) is scored from term ``j``'s own
vector, and reordering the bag reorders those logits and changes nothing else.
The global token carries the state's score, the step budget, the size of the
equation, the DOF, and :func:`discover_edit_env.residual_features` -- what the
equation still misses -- and it scores the moves that do not target a term
(STOP, ADD) and the value.

The logits come out in exactly :class:`discover_edit_env.EditSpec`'s action
layout; illegal moves are masked to -1e9, never to -inf, so a softmax can never
produce NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _encoder(d_model, n_heads, ff_dim, n_layers):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout=0.0,
                                       batch_first=True)
    # The nested-tensor fast path only runs in eval mode and changes the
    # numerics slightly; turning it off keeps the rollout's log-probs and the
    # update's identical.
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class EditPolicy(nn.Module):
    """``forward(tokens, term_mask, glob, dof) -> (logits[B, A], value[B])``.

    ``tokens``  (B, K, L) grammar token ids, ``EMPTY = n_tokens`` for an empty
                slot and ``PAD = n_tokens + 1`` after each term
    ``term_mask`` (B, K) True where a slot holds a term
    ``glob``    (B, G) global features, ``dof`` (B,) the DOF being edited
    """

    def __init__(self, n_tokens, max_terms, max_term_len, n_global_actions,
                 per_term, n_global_features, n_dof, d_model=64, n_heads=4,
                 ff_dim=128, n_layers=2):
        super().__init__()
        self.K, self.L = int(max_terms), int(max_term_len)
        self.n_global_actions = int(n_global_actions)
        self.per_term = int(per_term)
        self.EMPTY, self.PAD = int(n_tokens), int(n_tokens) + 1

        self.tok = nn.Embedding(n_tokens + 2, d_model, padding_idx=self.PAD)
        self.pos = nn.Embedding(self.L, d_model)
        self.term_enc = _encoder(d_model, n_heads, ff_dim, 1)
        self.glob_in = nn.Sequential(nn.Linear(n_global_features, d_model),
                                     nn.ReLU(), nn.Linear(d_model, d_model))
        self.dof_emb = nn.Embedding(max(int(n_dof), 1), d_model)
        self.bag_enc = _encoder(d_model, n_heads, ff_dim, n_layers)

        self.h_stop = nn.Linear(d_model, 1)
        self.h_glob = nn.Linear(d_model, self.n_global_actions)
        self.h_term = nn.Linear(d_model, self.per_term)
        self.v_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(),
                                    nn.Linear(d_model, 1))

    @classmethod
    def for_spec(cls, spec, n_tokens, n_global_features, **kw):
        return cls(n_tokens, spec.max_terms, spec.max_term_len, spec.n_global,
                   spec.per_term, n_global_features, spec.n_dof, **kw)

    def forward(self, tokens, term_mask, glob, dof):
        B, K, L = tokens.shape
        flat = tokens.reshape(B * K, L)
        pad = flat == self.PAD
        h = self.tok(flat) + self.pos.weight[:L]
        h = self.term_enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        h = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        h = h.reshape(B, K, -1)

        g = self.glob_in(glob) + self.dof_emb(dof)
        seq = torch.cat([g.unsqueeze(1), h], dim=1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
                         ~term_mask], dim=1)
        out = self.bag_enc(seq, src_key_padding_mask=kpm)
        g_out, t_out = out[:, 0], out[:, 1:]

        logits = torch.cat([self.h_stop(g_out), self.h_glob(g_out),
                            self.h_term(t_out).reshape(B, K * self.per_term)],
                           dim=1)
        return logits, self.v_head(g_out).squeeze(-1)


def masked_dist(logits, mask):
    """Categorical over the legal moves only."""
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, -1e9))
