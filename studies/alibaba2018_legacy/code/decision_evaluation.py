"""Shared deterministic IA-OSR policy evaluation; generalized to N domains.
Recovered from the previously audited Eagle pipeline; no Eagle inputs used.
"""
import math
import numpy as np
import pandas as pd
from ia_osr_experiment import Params
from ia_osr_psa_audit import PSASolver,clear_all
METHODS=('query_only','risk_query','screening_greedy','h1','psa1','exact')
FIELDS=('cost','success','empty_stop','miss','queries','refreshes','skips','model_stop_risk')


def codes(x, y):
    n=x.shape[1]
    weights=1<<np.arange(n)
    return x.astype(int)@weights+(1<<n)*(y.astype(int)@weights)


def independent_joint(par):
    n=par.n()
    mass=np.zeros(1<<(2*n))
    for k in range(len(mass)):
        pr=1.
        for i in range(n):
            x,y=bool(k&(1<<i)),bool(k&(1<<(i+n)))
            pr *= par.p[i]*(par.alpha[i] if y else 1-par.alpha[i]) if x else (1-par.p[i])*((1-par.beta[i]) if y else par.beta[i])
        mass[k]=pr
    assert abs(mass.sum()-1)<1e-10
    return mass


def make_tables(par):
    sol=PSASolver(par);n=par.n()
    tables=np.empty((len(METHODS),1<<(2*n),len(FIELDS)))
    for j,method in enumerate(METHODS):
        for k in range(tables.shape[1]):
            xx=tuple(bool(k&(1<<i)) for i in range(n))
            yy=tuple(bool(k&(1<<(i+n))) for i in range(n))
            tables[j,k]=episode(sol,xx,yy,method)
    mass=independent_joint(par)
    expected=mass@tables[:,:,0].T
    opt=sol.exact(sol.initial);psa=sol.psa_fast_cost(sol.initial)
    assert math.isclose(expected[METHODS.index('exact')],opt,abs_tol=1e-9)
    assert math.isclose(expected[METHODS.index('psa1')],psa,abs_tol=1e-9)
    assert opt<=psa+1e-9<=sol.HPSA1fast(sol.initial)+2e-9
    clear_all()
    return tables,expected

def calibrate(x: np.ndarray, y: np.ndarray):
    if len(x) < 100:
        raise ValueError('Fewer than 100 paired calibration bins; do not silently extrapolate.')
    pos = x.sum(axis=0)
    neg = len(x) - pos
    p = (pos + 1) / (len(x) + 2)  # explicitly an estimator, not a risk upper bound
    alpha = ((x & y).sum(axis=0) + 1) / (pos + 2)
    beta = ((~x & ~y).sum(axis=0) + 1) / (neg + 2)
    return tuple(p), tuple(alpha), tuple(beta)

def action(sol, state, method):
    if method == 'query_only':
        remaining = sol.unfinished(state)
        if not remaining:
            return 'STOP', -1
        return 'Q', min(remaining, key=lambda i: (sol.a.q[i] / max(sol.belief(i, state[i]), 1e-15), i))
    if sol.stop(state):
        return 'STOP', -1
    acts = sol.actions(state)
    if method in ('risk_query', 'screening_greedy'):
        skips = [a for a in acts if a[0] == 'SKIP']
        if skips:
            return max(skips, key=lambda a: (sol.a.q[a[1]], -a[1]))
        candidates = []
        for a in acts:
            op, i = a
            if op == 'Q':
                candidates.append((sol.a.q[i] / max(sol.belief(i, state[i]), 1e-15), 0, i, a))
            if op == 'R' and method == 'screening_greedy':
                rneg = 1 - (1 - sol.risk(state, True)) * (1 - sol.pm[i])
                if rneg <= sol.a.eps + 1e-12:
                    score = (sol.a.r[i] + sol.g[i] * sol.a.q[i]) / max(sol.a.p[i] * sol.a.alpha[i], 1e-15)
                    candidates.append((score, 1, i, a))
        return min(candidates)[-1]
    if method == 'h1':
        return sol.rollout_action(state, 'H1')
    if method == 'psa1':
        return sol.psa_fast_action(state)
    if method == 'exact':
        priority = {'STOP': 0, 'SKIP': 1, 'Q': 2, 'R': 3}
        return min(acts, key=lambda a: (round(sol.q_exact(state, a), 12), priority[a[0]], a[1]))
    raise ValueError(method)

def episode(sol, truth, observed, method):
    state = sol.initial
    cost = 0.0
    qn = rn = sn = 0
    for _ in range(4 * sol.n + 4):
        op, i = action(sol, state, method)
        if op == 'STOP':
            miss = int(any(truth[j] and k != 3 for j, k in enumerate(state)))
            risk = sol.risk(state)
            if method != 'query_only' and risk > sol.a.eps + 1e-10:
                raise AssertionError('Policy violates its own model STOP bound.')
            return np.array([cost, 0, 1, miss, qn, rn, sn, risk], dtype=float)
        if op == 'Q':
            qn += 1
            cost += sol.a.q[i]
            if truth[i]:
                return np.array([cost, 1, 0, 0, qn, rn, sn, 0], dtype=float)
            state = sol.set(state, i, 3)
        elif op == 'R':
            cost += sol.a.r[i]
            rn += 1
            state = sol.set(state, i, 1 if observed[i] else 2)
        elif op == 'SKIP':
            sn += 1
            state = sol.set(state, i, state[i] + 4)
    raise AssertionError('Policy failed to terminate.')

def correlations(values):
    matrix = pd.DataFrame(values.astype(float)).corr().to_numpy()
    return [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]

def weekly_block_bootstrap(daily: pd.DataFrame, numerator: str, denominator: str, draws=1000, seed=17):
    """Seven consecutive CALENDAR days per block, preserving days without data.

    Descriptive sensitivity to temporal dependence, not an assurance theorem.
    A missing denominator in a bootstrap replicate is discarded and counted.
    """
    daily = daily.sort_values('day')
    if len(daily) < 7:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    a = daily[numerator].to_numpy(float)
    b = daily[denominator].to_numpy(float)
    n = len(a)
    m = math.ceil(n / 7)
    start = rng.integers(0, n - 6, size=(draws, m))
    idx = (start[:, :, None] + np.arange(7)).reshape(draws, -1)[:, :n]
    den = b[idx].sum(axis=1)
    good = den > 0
    if not good.any():
        return np.nan, np.nan, 0
    values = a[idx][good].sum(axis=1) / den[good]
    lo, hi = np.quantile(values, [.025, .975])
    return float(lo), float(hi), int(good.sum())

