# Author: Can Zhang
# Toy recommender with item churn, retrained daily
# on a W-day window of its own logs; test designs.
import numpy as np

K, RHO, V, T = 200, 0.05, 5000, 14
EPS0 = 0.01
A_PRIOR, B_PRIOR = 1.0, 19.0
TAU_PC, GAMMA_PC = -0.004, 0.008
CELLS = [(28, 0.05), (28, 0.15), (56, 0.05), (56, 0.15)]
LAMBDAS = np.geomspace(0.5, 500, 60)


class Model:
    """Per-item clicks and imps over a W-day window."""

    def __init__(self, W):
        self.W, self.imp, self.clk, self.share = W, [], [], []

    def copy(self):
        m = Model(self.W)
        m.imp = [x.copy() for x in self.imp]
        m.clk = [x.copy() for x in self.clk]
        m.share = list(self.share)
        return m

    def add(self, imp, clk, share):
        self.imp.append(imp)
        self.clk.append(clk)
        self.share.append(share)
        if len(self.imp) > self.W:
            self.imp.pop(0), self.clk.pop(0), self.share.pop(0)

    def churn(self, slots):
        for a, b in zip(self.imp, self.clk):
            a[slots] = 0
            b[slots] = 0

    def scores(self):
        n = np.sum(self.imp, 0) if self.imp else np.zeros(K)
        c = np.sum(self.clk, 0) if self.clk else np.zeros(K)
        return (A_PRIOR + c) / (A_PRIOR + B_PRIOR + n)

    def s(self):
        """Treated share; empty days count as control."""
        return float(np.sum(self.share)) / self.W


class World:
    def __init__(self, W, rng, pc):
        self.W, self.rng, self.pc = W, rng, pc
        self.mu = rng.beta(A_PRIOR, B_PRIOR, K)

    def churn(self, models):
        slots = np.flatnonzero(self.rng.random(K) < RHO)
        self.mu[slots] = self.rng.beta(A_PRIOR, B_PRIOR, slots.size)
        for m in models:
            m.churn(slots)

    def serve(self, model, eps, n, arm, s=None):
        """eps-greedy on model scores -> imps, clicks."""
        n_exp = self.rng.binomial(n, eps)
        imp = self.rng.multinomial(n_exp, np.full(K, 1.0 / K))
        imp[np.argmax(model.scores())] += n - n_exp
        if self.pc:  # linear state model
            s = model.s() if s is None else s
            p = (0.05 + TAU_PC * arm + GAMMA_PC * s) * np.ones(K)
        else:
            p = self.mu
        return imp, self.rng.binomial(imp, p)

    def thin(self, imp, clk, m):
        """Subsample m impressions with their clicks."""
        if m >= imp.sum():
            return imp.copy(), clk.copy()
        sub = self.rng.multivariate_hypergeometric(imp, m)
        return sub, self.rng.hypergeometric(clk, imp - clk, sub)


def run_alone(W, eps, arm, rng, pc, days):
    """One arm on all traffic, own-log training."""
    w = World(W, rng, pc)
    m = Model(W)
    out = []
    for _ in range(days):
        w.churn([m])
        imp, clk = w.serve(m, eps, 2 * V, arm)
        out.append(clk.sum() / imp.sum())
        m.add(*w.thin(imp, clk, V), float(arm))
    return w, m, np.array(out)


def truth(W, eps1, pc, runs, cell):
    """Launch effect from long runs of each arm."""
    res = {}
    for arm, eps in ((0, EPS0), (1, eps1)):
        vals = []
        for r in range(runs):
            rng = np.random.default_rng([20261126, cell, arm, r])
            daily = run_alone(W, eps, arm, rng, pc, 3 * W + 60)[2]
            vals.append(daily[3 * W :].mean())
        res[arm] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(runs)))
    return res[1][0] - res[0][0], float(np.hypot(res[0][1], res[1][1]))


## the four designs


def pooled(w, m0, eps):
    m = m0.copy()
    ys = {0: [], 1: []}
    for _ in range(T):
        w.churn([m])
        logs = {a: w.serve(m, eps[a], V, a) for a in (0, 1)}
        for a in (0, 1):
            ys[a].append(logs[a][1].sum() / V)
        both_i = np.concatenate([logs[0][0], logs[1][0]])
        both_c = np.concatenate([logs[0][1], logs[1][1]])
        si, sc = w.thin(both_i, both_c, V)
        m.add(si[:K] + si[K:], sc[:K] + sc[K:], si[K:].sum() / V)
    return {"pooled": float(np.mean(ys[1]) - np.mean(ys[0]))}


def diverted(w, m0, eps):
    ms = {0: m0.copy(), 1: m0.copy()}
    ys = {0: [], 1: []}
    for _ in range(T):
        w.churn(list(ms.values()))
        for a in (0, 1):
            imp, clk = w.serve(ms[a], eps[a], V, a)
            ys[a].append(clk.sum() / V)
            ms[a].add(imp, clk, float(a))
    daily = np.array(ys[1]) - np.array(ys[0])
    return {
        "diverted": float(np.mean(ys[1]) - np.mean(ys[0])),
        "diverted_daily": [float(x) for x in daily],
    }


def switchback(w, m0, eps):
    m = m0.copy()
    order = w.rng.permutation([0] * (T // 2) + [1] * (T - T // 2))
    ys = {0: [], 1: []}
    for a in order:
        w.churn([m])
        imp, clk = w.serve(m, eps[a], 2 * V, a)
        ys[a].append(clk.sum() / (2 * V))
        m.add(*w.thin(imp, clk, V), float(a))
    return {"switchback": float(np.mean(ys[1]) - np.mean(ys[0]))}


def share(w, m0, eps):
    """Replicas w/ treated share f -> fit in s."""
    fs = (0.0, 0.5, 1.0)
    ms = {f: m0.copy() for f in fs}
    rows = []  # (arm, s, click rate)
    n_cell = 2 * V // 6
    zeros = (np.zeros(K, int), np.zeros(K, int))
    for _ in range(T):
        w.churn(list(ms.values()))
        logs = {}
        for f in fs:
            s_now = ms[f].s()
            for a in (0, 1):
                imp, clk = w.serve(ms[f], eps[a], n_cell, a)
                logs[(a, f)] = (imp, clk)
                rows.append((a, s_now, clk.sum() / n_cell))
        pool = {
            a: (sum(logs[(a, f)][0] for f in fs), sum(logs[(a, f)][1] for f in fs))
            for a in (0, 1)
        }
        for f in fs:
            n1 = int(round(f * V))
            i1, c1 = w.thin(*pool[1], n1) if n1 else zeros
            i0, c0 = w.thin(*pool[0], V - n1) if V - n1 else zeros
            ms[f].add(i0 + i1, c0 + c1, n1 / V)
    r = np.array(rows)
    fit = {}
    for a in (0, 1):
        x, y = r[r[:, 0] == a, 1], r[r[:, 0] == a, 2]
        fit[a] = np.polyfit(x, y, 1)  # slope, intercept
        if a == 1:
            q = np.polyfit(x, y, 2) if np.ptp(x) > 0 else None
    return {
        "share": float(fit[1][0] + fit[1][1] - fit[0][1]),
        "share_quadratic": float(np.polyval(q, 1.0) - fit[0][1])
        if q is not None
        else None,
        "share_smax": float(r[:, 1].max()),
    }


def test_designs(W, eps1, pc, cell, test):
    """One test of each design, shared pre-test."""
    rng0 = np.random.default_rng([20261125, cell, test, 0])
    w0, m0, _ = run_alone(W, EPS0, 0, rng0, pc, 3 * W)
    eps = {0: EPS0, 1: eps1}
    est = {}
    for di, design in enumerate((pooled, diverted, switchback, share)):
        w = World(W, np.random.default_rng([20261125, cell, test, di + 1]), pc)
        w.mu = w0.mu.copy()  # same items as the pre-test world
        est.update(design(w, m0, eps))
    return est


## variants of the fix


def share_world(w0, m0, W, eps1, rng, pc, days, traffic, short):
    """Two replicas (f = 0, 1) x serving arm.

    Rows: (arm, s, treated days in window, day, click rate).
    """
    w = World(W, rng, pc)
    w.mu = w0.mu.copy()
    m0 = m0.copy()  # churned below; keep the shared snapshot intact
    eps = {0: EPS0, 1: eps1}
    ms = {f: (Model(days) if short else m0.copy()) for f in (0, 1)}
    rows, n_cell = [], traffic // 2
    for day in range(days):
        w.churn(list(ms.values()) + [m0])
        logs = {}
        for f in (0, 1):
            m = ms[f] if ms[f].imp else m0  # day 1 of SW: pre-test model
            s_now = m.s() if not short else float(f if ms[f].imp else 0.0)
            n_tr = s_now * (len(ms[f].imp) if short else W)
            for a in (0, 1):
                imp, clk = w.serve(m, eps[a], n_cell, a, s_now)
                logs[(a, f)] = (imp, clk)
                rows.append((a, s_now, n_tr, day, clk.sum() / n_cell))
        pool = {
            a: (logs[(a, 0)][0] + logs[(a, 1)][0], logs[(a, 0)][1] + logs[(a, 1)][1])
            for a in (0, 1)
        }
        for f in (0, 1):
            # trains on V a day whatever the traffic
            ms[f].add(*w.thin(*pool[f], V), float(f))
    return np.array(rows)


def linear(r):
    fit = {a: np.polyfit(r[r[:, 0] == a, 1], r[r[:, 0] == a, 4], 1) for a in (0, 1)}
    return float(fit[1][0] + fit[1][1] - fit[0][1]), fit


def learning_curve(r, W):
    _, fit = linear(r)
    x, y = r[r[:, 0] == 1, 2], r[r[:, 0] == 1, 4]
    best = None
    for lam in LAMBDAS:
        z = 1 - np.exp(-x / lam)
        X = np.column_stack([np.ones_like(z), z])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((y - X @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, lam, coef)
    _, lam, (a1, b) = best
    return float(a1 + b * (1 - np.exp(-W / lam)) - fit[0][1]), float(lam)


def short_window(r, days):
    last = r[:, 3] >= days - 7
    y11 = r[last & (r[:, 0] == 1) & (r[:, 1] == 1), 4].mean()
    y00 = r[last & (r[:, 0] == 0) & (r[:, 1] == 0), 4].mean()
    return float(y11 - y00)


def test_variants(W, eps1, pc, cell, test, extra_traffic=False):
    """One simulated test of each variant of the fix."""
    rng0 = np.random.default_rng([20261127, cell, test, 0])
    w0, m0, _ = run_alone(W, EPS0, 0, rng0, pc, 3 * W)
    specs = [("S2", 14, V, False), ("S2L", 28, V, False), ("SW", 14, V, True)]
    if extra_traffic:
        specs = [("S2x10", 14, 10 * V, False)]
    out = {}
    for vi, (name, days, traffic, short) in enumerate(specs):
        rng = np.random.default_rng([20261127, cell, test, vi + 1])
        r = share_world(w0, m0, W, eps1, rng, pc, days, traffic, short)
        if short:
            out["short_window"] = short_window(r, days)
        else:
            out[f"{name}_linear"] = linear(r)[0]
            if not pc:
                out[f"{name}_lc"], out[f"{name}_lambda"] = learning_curve(r, W)
    return out
