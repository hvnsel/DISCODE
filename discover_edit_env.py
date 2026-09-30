"""
discover_edit_env.py
====================

PROTOTYPE ADD-ON.  The equation search as a sequence of EDITS: an agent holds
one DOF's current equation and changes it a move at a time -- add a term, raise
something to a power, multiply a term by a variable, delete a term -- and every
move is scored.  It is trained with PPO in :mod:`discover_edit_train`.

Nothing here changes the existing pipeline.  The grammar, the constant fitter
and the reward all come from :mod:`discover_core`, used unchanged.

Why edits
---------
The term-bag policy (:mod:`discover_policy`) writes each equation from scratch
and learns from one number at the end.  An editor is paid for every move, so
credit assignment is local; it sees the equation it is working on; and it can
be shown what that equation still gets wrong (:func:`residual_features`), which
the from-scratch policy never sees.

State
-----
A bag of terms for one DOF, summed -- the representation the term-bag policy
already uses, so every state is a tau that :func:`discover_core.assemble_terms`
builds and :func:`discover_core.optimise_consts_energy` fits.  Terms are built
only by the moves below.  In the default closed-form grammar every term is a
product of leaves and powers of leaves, so every fit takes the fast closed-form
path (except powers of a directional leaf, which are nonlinear by nature).

Moves
-----
    STOP            end the episode (not allowed on an empty equation)
    ADD(v)          append the leaf v as a new term            x
    ADD(op, v)      append op(v) as a new term                 x^3
    WRAP(j, op)     apply op to the LAST factor of term j      x*y  ->  x*y^2
    MUL(j, v)       multiply term j by the leaf v              x^2  ->  x^2*xdot
    DELETE(j)       remove term j
    WRAP_ALL(op)    wrap the whole equation, op(t1 + ... + tn)   (open grammar)

WRAP acts on the last factor so that it means one thing in both grammars: a
single-factor term is its own last factor, so ``WRAP(j, intpower)`` on ``x``
gives ``x^n``, and on a product it powers what was multiplied in last.

A move is legal only if every resulting term is one the term grammar could have
written (:func:`discover_core.get_valid_tokens` in term mode), the bag stays
within ``max_terms`` / ``max_term_len``, and no two terms are the same up to
factor order: a duplicate is a collinear column that costs a constant and fits
nothing.  The closed-form grammar also requires every term to be separable
(:func:`_separable`), which is what actually keeps its fits fast.  A
``const`` is only ever a term of its own (an offset) or part of a WRAP_ALL sum;
``c * x`` or ``c^n`` would be a second amplitude on a term that has one.

Reward
------
    score(s) = -log(E(s)) - lam * n_consts(s)            r = 1 / (1 + E)

``E`` is the residual behind the standard reward, so ``-log E`` is
:func:`discover_core.log_residual_score` in closed form.  It spreads the near-1
end of ``r``, where every useful comparison lives: on duffing the linear part
scores 1.33 and the truth 10.52.  Each move is paid ``score(s') - score(s)``,
which telescopes, so with gamma = 1 an episode's return is its final score
minus its starting score -- maximising return IS maximising the score of the
equation the agent stops on.

``lam`` is charged per FITTED CONSTANT, not per token.  ``power(x)`` and
``intpower(x)`` are the same two tokens, but power's extra amplitude and
continuous exponent buy 0.14 on duffing, so a token count would pay the agent
to use the non-standard power.  The default ``lam = 0.5`` is set by that
margin, which depends on the data: on duffing's standard record the intpower
truth beats the power variant at any ``lam`` above 0.14, but on a 2-trial,
600-point record power buys 0.32 and needs ``lam`` above that.  Junk terms buy
0.04 to 0.39 for 3 to 4 constants and a real term 2 to 10, so the registry
truths win by a clear margin at 0.5.  The price is a deeper dip on paths whose
first moves only pay later -- van der Pol's ``x^2 * xdot`` costs two moves
before it pays.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

import discover_core as dc


# ``E`` is clipped before the log: a perfect fit must not score infinity, and a
# failed fit (``r = 0``) must cost a finite, strongly negative amount.
E_FLOOR = 1e-9
E_CEIL  = 1e6

# ``Scored.tau`` / ``consts`` are in CANONICAL term order (sorted), which is the
# order the bag was fitted in; the env's own slot order is irrelevant to scoring.
Scored = namedtuple('Scored', 'E neg_log_E n_consts tau consts feats')


def bag_from_tau(tau):
    """A flat tau (e.g. a registry truth) as a bag of term tuples."""
    return tuple(tuple(t) for t in dc.split_terms(list(tau)))


def _factors(term):
    """The factors of a term: a ``mul``'s children, or the term itself."""
    term = tuple(term)
    if not term or term[0] != 'mul':
        return [term]
    out, pos = [], 1
    while pos < len(term) and term[pos] != 'end':
        end = dc._subtree_end(list(term), pos)
        out.append(term[pos:end])
        pos = end
    return out


def _join(factors):
    """Inverse of :func:`_factors`: one factor is the term, several a ``mul``."""
    if len(factors) == 1:
        return tuple(factors[0])
    return ('mul',) + tuple(tok for f in factors for tok in f) + ('end',)


def _signature(term):
    """A term up to the order of its factors: ``x*y`` and ``y*x`` fit the same
    column, so the bag must not hold both."""
    return tuple(sorted(_factors(term)))


def canonical(terms):
    """The bag with its terms, and each product's factors, in sorted order:
    one key per FUNCTION, so the cache fits ``x*x*xdot`` once however its
    factors were multiplied in, and the hall of fame lists it once."""
    return tuple(sorted(_join(sorted(_factors(t))) for t in terms))


def _separable(term):
    """Would the closed-form fitter take this term?  Every monomial of its
    expansion needs an amplitude no other monomial shares -- the same test
    :func:`discover_core._fit_consts_linear` applies.  It fails for a product
    of two two-amplitude factors: ``power(x) * power(xdot)`` expands to four
    monomials over two pairs of amplitudes, and fell to the nonlinear
    optimiser at 15-27 s a fit against 0.25 s for a separable five-term bag.
    Terms never share slots, so a bag is closed-form iff each term is."""
    try:
        monos = dc._expand_monomials(dc._parse_tree(list(term))[0])
    except Exception:
        return False
    if not monos or len(monos) > dc.MAX_MONOMIALS:
        return False
    amps = [m['amp'] for m in monos]
    for j, a in enumerate(amps):
        others = set()
        for k, b in enumerate(amps):
            if k != j:
                others |= b
        if not (a - others):
            return False
    return True


class EditSpec:
    """The move set for the CURRENTLY CONFIGURED grammar, and its legality.

    Build it after :func:`discover_core.configure_grammar`: the leaves and
    operators it offers are read from the live token table.

    Action layout -- the policy assembles its logits in exactly this order::

        0                       STOP
        1 .. n_global           ADD(v), ADD(op, v), [WRAP_ALL(op)]
        then for each slot j:   WRAP(j, op) for op in ops,
                                MUL(j, v)   for v in leaves,
                                DELETE(j)                   (per_term per slot)
    """

    def __init__(self, max_terms=6, max_term_len=8, max_steps=10, lam=0.5,
                 closed_form=True):
        self.max_terms    = int(max_terms)
        self.max_term_len = int(max_term_len)
        self.max_steps    = int(max_steps)
        self.lam          = float(lam)
        self.closed_form  = bool(closed_form)
        self.n_dof        = dc.N_DOF

        dir_leaves = [t for t in dc.DIR_LEAVES if t in dc.DIR_LEAF_SET]
        self.leaves     = list(dc.VARIABLES) + dir_leaves     # factors of a term
        self.add_leaves = self.leaves + list(dc.CONSTANTS)    # ADD(v) also takes const
        self.ops        = list(dc.UNARY_OPS) + list(dc.TRANSCENDENTAL_OPS)

        actions = [('stop',)]
        actions += [('add', v) for v in self.add_leaves]
        actions += [('addop', op, v) for op in self.ops for v in self.leaves]
        if not self.closed_form:
            actions += [('wrapall', op) for op in self.ops]
        self.n_global = len(actions) - 1
        for j in range(self.max_terms):
            actions += [('wrap', j, op) for op in self.ops]
            actions += [('mul', j, v) for v in self.leaves]
            actions += [('del', j)]
        self.actions   = actions
        self.n_actions = len(actions)
        self.per_term  = len(self.ops) + len(self.leaves) + 1
        self.STOP      = 0

        self._term_ok_cache = {}
        self._mask_cache    = {}

    # ── legality ────────────────────────────────────────────────────────────
    def term_ok(self, term):
        """Could the term grammar have written ``term``?  Cached per term."""
        term = tuple(term)
        hit = self._term_ok_cache.get(term)
        if hit is not None:
            return hit
        ok = 0 < len(term) <= self.max_term_len and dc.is_complete(list(term))
        if ok:
            for i, tok in enumerate(term):
                m = dc.get_valid_tokens(
                    list(term[:i]), i, self.max_term_len, depth_offset=1,
                    min_len=1, allow_leaf_at_zero=True, forbid_at_root=('add',),
                    power_child_vars_only=self.closed_form)
                if m[dc.tok2idx(tok)] <= 0:
                    ok = False
                    break
        # The closed-form grammar promises the fast fit.  A directional leaf is
        # exempt: a power of one is nonlinear by nature, and that is the point.
        if (ok and self.closed_form and not _separable(term)
                and not any(tok in dc.DIR_LEAF_SET for tok in term)):
            ok = False
        self._term_ok_cache[term] = ok
        return ok

    def apply(self, terms, a):
        """The bag after action ``a``, or ``None`` if the move is illegal.

        STOP returns the bag unchanged (legal on any non-empty bag).
        """
        terms = tuple(tuple(t) for t in terms)
        act = self.actions[a]
        kind = act[0]
        n = len(terms)
        if kind == 'stop':
            return terms if n > 0 else None
        if kind == 'add':
            if n >= self.max_terms:
                return None
            new = terms + ((act[1],),)
        elif kind == 'addop':
            if n >= self.max_terms:
                return None
            new = terms + ((act[1], act[2]),)
        elif kind == 'wrapall':
            if n == 0:
                return None
            if n == 1:
                t = (act[1],) + terms[0]
            else:
                t = (act[1], 'add') + tuple(tok for x in terms for tok in x) + ('end',)
            new = (t,)
        else:
            j = act[1]
            if j >= n:
                return None
            term = terms[j]
            if kind == 'del':
                new = terms[:j] + terms[j + 1:]
            elif kind == 'wrap':
                f = _factors(term)
                if f[-1] == ('const',):
                    return None
                new_term = _join(f[:-1] + [(act[2],) + f[-1]])
                new = terms[:j] + (new_term,) + terms[j + 1:]
            elif kind == 'mul':
                if term == ('const',):
                    return None
                new_term = _join(_factors(term) + [(act[2],)])
                new = terms[:j] + (new_term,) + terms[j + 1:]
            else:
                raise ValueError(f"unknown action {act}")
        if len({_signature(t) for t in new}) != len(new):    # duplicate term
            return None
        if kind != 'del' and not all(self.term_ok(t) for t in new):
            return None
        return new

    def legal_mask(self, terms):
        """Boolean mask over ``actions`` for the bag ``terms`` (cached)."""
        key = tuple(tuple(t) for t in terms)
        hit = self._mask_cache.get(key)
        if hit is None:
            hit = np.array([self.apply(key, a) is not None
                            for a in range(self.n_actions)], dtype=bool)
            if len(self._mask_cache) > 200_000:
                self._mask_cache.clear()
            self._mask_cache[key] = hit
        return hit

    def reachable(self, terms):
        """Is every term of this bag writable by the moves (same legality)?"""
        terms = tuple(tuple(t) for t in terms)
        return (len(terms) <= self.max_terms
                and len({_signature(t) for t in terms}) == len(terms)
                and all(self.term_ok(t) for t in terms))

    def score(self, scored):
        """``-log E - lam * n_consts`` for a :class:`Scored` record."""
        return scored.neg_log_E - self.lam * scored.n_consts


# ── what the equation still misses ──────────────────────────────────────────
def n_residual_features(n_vars):
    """Shapes offered per state: u, u^2, u^3 per channel, u_i*u_j per pair,
    and u_i^2*u_j per ordered pair (van der Pol's x^2*xdot is one)."""
    return 3 * n_vars + n_vars * (n_vars - 1) // 2 + n_vars * (n_vars - 1)


def _shape_rows(F):
    nv = F.shape[0]
    rows = [F, F ** 2, F ** 3]
    pairs = [F[i] * F[j] for i in range(nv) for j in range(i + 1, nv)]
    cross = [F[i] ** 2 * F[j] for i in range(nv) for j in range(nv) if i != j]
    if pairs:
        rows.append(np.array(pairs))
    if cross:
        rows.append(np.array(cross))
    return np.vstack(rows)


def residual_features(tau, consts, dof):
    """How much of the DOF's measured acceleration each simple shape could
    still explain -- the agent's view of the data.

    Matching pursuit, one step: the acceleration residual and every candidate
    shape are projected off the span of the equation's own terms (plus an
    offset), and each feature is ``|<r_perp, s_perp>| / (|s_perp| |a - mean a|)``:
    the fraction of the acceleration's RMS that adding that shape could remove
    beyond what the current terms can.  Scale-aware, so it goes to ~0 at a
    good fit -- a plain correlation does not: on van der Pol's fitted truth it
    reported 0.995 against ``x``, a shape the equation already contains, from a
    misfit of 1e-4.  All zero if the record carries no measured acceleration.
    """
    nv = dc.N_VARS
    zeros = np.zeros(n_residual_features(nv))
    if dc.NORM_STATS is None or not dc.RAW_TRAJECTORIES:
        return zeros
    X_mean, X_std, y_mean, y_std = dc.NORM_STATS
    terms = dc.split_terms(list(tau))
    offs = np.cumsum([0] + [dc.count_total_consts(t) for t in terms])
    fns = [dc.compile_to_numpy(t, consts[offs[k]:offs[k + 1]])
           for k, t in enumerate(terms)]
    if any(f is None for f in fns):
        return zeros
    acc_all, cols_all, feats_all = [], [], []
    for traj in dc.RAW_TRAJECTORIES[:dc.MAX_TRAJ]:
        acc = dc._slice_accs(traj, None)
        if acc is None:
            return zeros
        _t, st = dc._slice_traj(traj, None)
        feats = (st - X_mean[:, None]) / X_std[:, None]
        n = feats.shape[1]
        with np.errstate(invalid='ignore', over='ignore'):
            cols = [np.broadcast_to(np.asarray(f(*feats), dtype=float), (n,))
                    for f in fns]
        acc_all.append(acc[dof])
        cols_all.append(np.array(cols))
        feats_all.append(feats)
    a = np.concatenate(acc_all)
    T = np.concatenate(cols_all, axis=1)             # (n_terms, n) term values
    F = np.concatenate(feats_all, axis=1)
    pred = y_mean[dof] + y_std[dof] * T.sum(axis=0)
    r = a - pred
    C = np.vstack([np.ones_like(a), T]).T            # span already in the model
    S = _shape_rows(F).T                             # (n, n_shapes)
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(r))):
        return zeros
    r_p = r - C @ np.linalg.lstsq(C, r, rcond=None)[0]
    S_p = S - C @ np.linalg.lstsq(C, S, rcond=None)[0]
    a_c = np.linalg.norm(a - a.mean()) + 1e-12
    out = np.abs(S_p.T @ r_p) / ((np.linalg.norm(S_p, axis=0) + 1e-12) * a_c)
    return np.where(np.isfinite(out), out, 0.0)


# ── scoring ─────────────────────────────────────────────────────────────────
def score_job(job):
    """Fit and score one bag for one DOF.  Module-level so a process pool can
    run it; the worker must have been initialised with
    :func:`discover_core.init_energy_worker` (same grammar, same data).

    ``job = (dof, canonical_bag, want_features)``.  The empty bag is the mean
    model -- normalised prediction 0, i.e. the mean measured acceleration.
    """
    dof, canon, want_feats = job
    if canon:
        tau = dc.assemble_terms([list(t) for t in canon])
        consts = list(dc.optimise_consts_energy(tau, dof))
        n_consts = dc.count_total_consts(tau)
    else:
        tau, consts, n_consts = ['const'], [0.0], 0
    exprs = [None] * dc.N_DOF
    exprs[dof] = (tau, consts)
    r = dc.energy_reward(exprs)
    E = (1.0 / r - 1.0) if r > 0.0 else np.inf
    feats = residual_features(tau, consts, dof) if want_feats else None
    return dof, canon, float(E), int(n_consts), tau, consts, feats


class Scorer:
    """Fits and scores bags, cached by ``(dof, canonical bag)``.

    The same bag is reached by many move orders and in many episodes, and
    ``x + xdot`` is visited in nearly every one; the cache is what makes paying
    for every move affordable.  ``pool`` is an optional ProcessPoolExecutor
    whose workers ran :func:`discover_core.init_energy_worker`.
    """

    def __init__(self, spec, pool=None, residual_features=True):
        self.spec  = spec
        self.pool  = pool
        self.feats = bool(residual_features)
        self.cache = {}
        self.n_fits = 0

    def score_many(self, items):
        """Score every ``(dof, terms)`` not already cached; returns how many
        fits that took."""
        jobs, seen = [], set()
        for dof, terms in items:
            key = (int(dof), canonical(terms))
            if key in self.cache or key in seen:
                continue
            seen.add(key)
            jobs.append((key[0], key[1], self.feats))
        if not jobs:
            return 0
        if self.pool is not None and len(jobs) > 1:
            # chunksize 1: fit times vary by 100x, and one chunk of slow fits
            # would idle the other workers
            results = list(self.pool.map(score_job, jobs, chunksize=1))
        else:
            results = [score_job(j) for j in jobs]
        for dof, canon, E, n_c, tau, consts, feats in results:
            neg = -float(np.log(np.clip(E, E_FLOOR, E_CEIL)))
            self.cache[(dof, canon)] = Scored(E, neg, n_c, tau, consts, feats)
        self.n_fits += len(jobs)
        return len(jobs)

    def get(self, dof, terms):
        key = (int(dof), canonical(terms))
        if key not in self.cache:
            self.score_many([(dof, terms)])
        return self.cache[key]

    def value(self, dof, terms):
        """The state's score, ``-log E - lam * n_consts``."""
        return self.spec.score(self.get(dof, terms))


# ── observations ────────────────────────────────────────────────────────────
N_SCALAR_FEATURES = 5


def n_global_features(residual=True):
    """Width of ``encode``'s ``glob`` block for the configured grammar."""
    return N_SCALAR_FEATURES + (n_residual_features(dc.N_VARS) if residual else 0)


def encode(spec, scorer, states):
    """Batch observation for ``states = [(dof, terms, step), ...]``.

    Returns numpy arrays ``(tokens[B,K,L], term_mask[B,K], glob[B,G],
    dof[B], action_mask[B,A])``.  Token ids are the grammar's; an empty slot is
    ``[EMPTY, PAD, ...]`` (so no row is all padding) with ``EMPTY =
    N_TOKENS`` and ``PAD = N_TOKENS + 1``.  Every state must already be scored.
    """
    K, L = spec.max_terms, spec.max_term_len
    EMPTY, PAD = dc.N_TOKENS, dc.N_TOKENS + 1
    B = len(states)
    tokens = np.full((B, K, L), PAD, dtype=np.int64)
    tokens[:, :, 0] = EMPTY
    term_mask = np.zeros((B, K), dtype=bool)
    n_res = n_residual_features(dc.N_VARS) if scorer.feats else 0
    glob = np.zeros((B, N_SCALAR_FEATURES + n_res), dtype=np.float32)
    dofs = np.zeros(B, dtype=np.int64)
    amask = np.zeros((B, spec.n_actions), dtype=bool)
    for b, (dof, terms, step) in enumerate(states):
        for j, t in enumerate(terms):
            tokens[b, j, :len(t)] = [dc.tok2idx(tok) for tok in t]
            term_mask[b, j] = True
        s = scorer.get(dof, terms)
        glob[b, 0] = s.neg_log_E / 10.0
        glob[b, 1] = spec.score(s) / 10.0
        glob[b, 2] = step / max(spec.max_steps, 1)
        glob[b, 3] = len(terms) / spec.max_terms
        glob[b, 4] = s.n_consts / 20.0
        if n_res and s.feats is not None:
            glob[b, N_SCALAR_FEATURES:] = s.feats
        dofs[b] = dof
        amask[b] = spec.legal_mask(terms)
    return tokens, term_mask, glob, dofs, amask
