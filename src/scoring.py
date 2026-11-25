# Author: Can Zhang
# Scores by the rules set before the research runs;
# same text as the research score files.
import numpy as np

T = 14
DESIGNS = ["pooled", "diverted", "switchback", "share"]


def boot_ci(x, rng, b=2000):
    x = np.asarray(x)
    meds = np.median(x[rng.integers(0, x.size, (b, x.size))], 1)
    return np.percentile(meds, [2.5, 97.5])


def score_pc(d):
    c = d["cells"][0]
    tau, gamma = -0.004, 0.008
    delta, W = c["delta"], c["W"]
    lines = []
    for name in DESIGNS:
        x = np.array([t[name] for t in c["tests"]])
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(x.size)
        lines.append(f"{name}: mean {m:.5f} (SE {se:.5f})")
    share = np.array([t["share"] for t in c["tests"]])
    err, se = share.mean() - delta, share.std(ddof=1) / np.sqrt(share.size)
    c1 = abs(err) <= 3 * se and abs(err) <= 0.05 * abs(delta)
    kappa_pred = np.mean(np.arange(T)) / W  # window share on day d: (d - 1) / W
    kap = (np.mean([t["diverted"] for t in c["tests"]]) - tau) / gamma
    c2 = abs(kap - kappa_pred) <= 0.05
    c3 = all(
        abs(np.mean([t[n] for t in c["tests"]]) - tau) <= 0.1 * abs(gamma)
        for n in ("pooled", "switchback")
    )
    ok = c1 and c2 and c3
    lines += [
        f"share error {err:.6f} (SE {se:.6f}, 5% of Delta = {0.05 * abs(delta):.6f}): {'ok' if c1 else 'FAIL'}",
        f"diverted kappa {kap:.3f} vs predicted {kappa_pred:.3f}: {'ok' if c2 else 'FAIL'}",
        f"pooled and switchback within 0.1|gamma| of tau: {'ok' if c3 else 'FAIL'}",
        f"PC {'PASS' if ok else 'FAIL'}",
    ]
    return "\n".join(lines)


def score_designs(d):
    rng = np.random.default_rng(1)
    lines, scored = [], []
    for c in d["cells"]:
        delta, se = c["delta"], c["delta_se"]
        head = f"W={c['W']} eps1={c['eps1']}: Delta {delta:.5f} (SE {se:.5f})"
        if abs(delta) < 3 * se:
            lines.append(head + " -- not scored (|Delta| < 3 SE)")
            continue
        row = {}
        for name in DESIGNS + ["share_quadratic"]:
            x = np.array([t[name] for t in c["tests"] if t.get(name) is not None])
            rel = (x - delta) / abs(delta)
            lo, hi = boot_ci(rel, rng)
            row[name] = np.median(rel)
            head += f"\n  {name:16s} median rel. error {np.median(rel):+.3f} ({lo:+.3f}, {hi:+.3f}); estimate sd {x.std():.5f}"
        smax = np.mean([t["share_smax"] for t in c["tests"]])
        lines.append(head + f"\n  share design s_max {smax:.3f}")
        scored.append(row)
    if len(scored) < 3:
        return "\n".join(lines + ["KT1 NOT ASSESSABLE (fewer than 3 scored cells)"])
    std = ["pooled", "diverted", "switchback"]
    good = all(abs(r["share"]) <= 0.25 for r in scored)
    problem = sum(all(abs(r[n]) >= 0.5 for n in std) for r in scored) >= len(scored) / 2
    bad = sum(abs(r["share"]) > 0.5 for r in scored) >= 2
    never_best = not any(abs(r["share"]) < min(abs(r[n]) for n in std) for r in scored)
    verdict = (
        "PASS" if good and problem else "KILL" if bad or never_best else "MARGINAL"
    )
    lines.append(
        f"share within 0.25 in every scored cell: {good}; every standard design >= 0.5 in at least half: "
        f"{problem}; share > 0.5 in two or more cells: {bad}; share never best: {never_best}"
    )
    lines.append(f"KT1 {verdict}")
    return "\n".join(lines)


def score_pc_variants(d):
    c = d["cells"][0]
    out, ok = [], True
    for name in ("S2_linear", "S2L_linear"):
        x = np.array([t[name] for t in c["tests"]])
        err, se = x.mean() - c["delta"], x.std(ddof=1) / np.sqrt(x.size)
        good = abs(err) <= 3 * se
        ok &= good
        out.append(
            f"{name}: mean error {err:+.6f} (SE {se:.6f}, {abs(err) / se:.2f} SE): {'ok' if good else 'FAIL'}"
        )
    x = np.array([t["short_window"] for t in c["tests"]])
    out.append(f"short_window (reported): mean error {x.mean() - c['delta']:+.6f}")
    out.append(f"PC {'PASS' if ok else 'FAIL'}")
    return "\n".join(out)


def score_variants(d):
    rng = np.random.default_rng(2)
    est = ["S2_linear", "S2L_linear", "S2_lc", "short_window", "S2L_lc"]
    lines, rows = [], []
    for c in d["cells"]:
        delta, se = c["delta"], c["delta_se"]
        head = f"W={c['W']} eps1={c['eps1']}: Delta {delta:.5f} (SE {se:.5f})"
        if abs(delta) < 3 * se:
            lines.append(head + " -- not scored")
            continue
        row = {}
        for name in est:
            x = np.array([t[name] for t in c["tests"]])
            rel = (x - delta) / abs(delta)
            lo, hi = boot_ci(rel, rng)
            row[name] = (float(np.median(rel)), float(x.std() / abs(delta)))
            head += f"\n  {name:13s} median rel. error {np.median(rel):+.3f} ({lo:+.3f}, {hi:+.3f}); noise ratio {row[name][1]:.2f}"
        lam = np.array([t["S2_lambda"] for t in c["tests"]])
        head += f"\n  S2 learning-curve lambda: median {np.median(lam):.1f} days, share at grid ends {np.mean((lam <= 0.5) | (lam >= 500)):.2f}"
        if c.get("extra"):
            x = np.array([t["S2x10_linear"] for t in c["extra"]])
            head += (
                f"\n  reported: S2 with 10x traffic, median rel. error {np.median((x - delta) / abs(delta)):+.3f}, "
                f"noise ratio {x.std() / abs(delta):.2f}"
            )
        lines.append(head)
        rows.append(row)
    if len(rows) < 3:
        return "\n".join(lines + ["NOT ASSESSABLE (fewer than 3 scored cells)"])
    kt2 = {}
    for name in ("S2_linear", "S2L_linear"):
        kt2[name] = (
            all(r[name][1] <= 1.0 for r in rows)
            and sum(abs(r[name][0]) <= 0.25 for r in rows) >= 3,
            all(r[name][1] > 2.0 for r in rows),
        )
    v2 = (
        "PASS"
        if any(p for p, _ in kt2.values())
        else "KILL"
        if all(k for _, k in kt2.values())
        else "MARGINAL"
    )
    lines.append(f"KT2 per variant (pass, kill): {kt2} -> KT2 {v2}")
    for name, label in (
        ("S2_lc", "KT3 primary (learning curve)"),
        ("short_window", "KT3 secondary (short window)"),
    ):
        good = all(abs(r[name][0]) <= 0.25 for r in rows)
        bad = sum(abs(r[name][0]) > 0.5 for r in rows) >= 2
        lines.append(f"{label}: {'PASS' if good else 'KILL' if bad else 'MARGINAL'}")
    return "\n".join(lines)
