# Author: Can Zhang
# Reruns the controls and studies (research seeds)
# -> results/*.json and results/*_score.txt
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # before numpy loads

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import recsim as rs  # noqa: E402
import scoring  # noqa: E402

N_PC = 2000  # research run used 10,000 for the designs' control
N_TESTS = 100


def run_cells(one_test, cells, n, pc, truths=None, extra=None):
    res = []
    for W, eps1 in cells:
        cell = rs.CELLS.index((W, eps1))
        delta, se = (rs.TAU_PC + rs.GAMMA_PC, 0.0) if pc else truths[cell]
        c = dict(W=W, eps1=eps1, delta=delta, delta_se=se)
        c["tests"] = [one_test(W, eps1, pc, cell, t) for t in range(n)]
        if extra is not None:  # 10x traffic, first cell only
            c["extra"] = (
                [extra(W, eps1, cell, t) for t in range(n)] if cell == 0 else []
            )
        res.append(c)
    return res


def ten_x(W, eps1, cell, t):
    return rs.test_variants(W, eps1, False, cell, t, extra_traffic=True)


def save(name, n, cells, score):
    d = dict(n_tests=n, cells=cells)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(d), newline="\n")
    (ROOT / "results" / f"{name}_score.txt").write_text(score(d) + "\n", newline="\n")


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    t0 = time.time()

    def done(step):
        print(f"{step}: {time.time() - t0:.0f} s", flush=True)

    pc_cell = rs.CELLS[:1]
    res = run_cells(rs.test_designs, pc_cell, N_PC, True)
    save("positive_control", N_PC, res, scoring.score_pc)
    done("positive control, four designs")
    res = run_cells(rs.test_variants, pc_cell, N_PC, True)
    save("positive_control_variants", N_PC, res, scoring.score_pc_variants)
    done("positive control, variants")

    # one truth per cell, shared by both studies
    truths = {i: rs.truth(W, e, False, 200, i) for i, (W, e) in enumerate(rs.CELLS)}
    done("launch effects")
    res = run_cells(rs.test_designs, rs.CELLS, N_TESTS, False, truths)
    save("designs", N_TESTS, res, scoring.score_designs)
    done("four designs")
    res = run_cells(rs.test_variants, rs.CELLS, N_TESTS, False, truths, ten_x)
    save("fix_variants", N_TESTS, res, scoring.score_variants)
    done("variants of the fix")


if __name__ == "__main__":
    main()
