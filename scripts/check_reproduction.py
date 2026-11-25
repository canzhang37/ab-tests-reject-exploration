# Author: Can Zhang
# P1 and P2: results/ vs the research run's result
# files (folder given as the only argument)
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-7  # identical to 6 decimals


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}/{i}")
    else:
        yield path, x


def gap(x, y):
    if x is None or y is None:  # quadratic fit not always possible
        return 0.0 if x is y else float("inf")
    return abs(x - y)


def numbers(d, n):
    out = {}
    for i, c in enumerate(d["cells"]):
        c = dict(c, tests=c["tests"][:n], extra=c.get("extra", []))
        out.update(leaves(c, f"/{i}"))
    return out


def same_numbers(name, src_file):
    """Stored numbers vs the source's first n tests."""
    ours = json.loads((ROOT / "results" / f"{name}.json").read_text())
    n = ours["n_tests"]
    a, b = numbers(ours, n), numbers(json.loads(src_file.read_text()), n)
    if a.keys() != b.keys():
        print(
            f"  numbers: fields differ from the source ({len(a.keys() ^ b.keys())} paths)"
        )
        return False
    diff = max(gap(a[k], b[k]) for k in a)
    print(
        f"  numbers: {len(a):,} compared with the source's first {n:,} tests, max |diff| {diff:.1e}"
    )
    return diff < TOL


def score_text(name, src_score=None, expect=()):
    text = (ROOT / "results" / f"{name}_score.txt").read_text().strip()
    print("  " + text.replace("\n", "\n  "))
    ok = all(e in text.splitlines() for e in expect)
    if src_score is not None:
        same = text == src_score.read_text().replace("\r\n", "\n").strip()
        print(f"  score file identical to the source's: {same}")
        ok &= same
    return ok


def main():
    src = Path(sys.argv[1])

    def find(pattern):
        return next(src.glob(pattern))

    print("P1 (a) positive control, four designs")
    p1 = same_numbers("positive_control", find("pc_*_amend1/pc.json"))
    p1 &= score_text("positive_control", expect=["PC PASS"])
    print("P1 (b) positive control, variants of the fix")
    p1 &= same_numbers("positive_control_variants", find("pc23_*/pc.json"))
    p1 &= score_text("positive_control_variants", find("pc23_*/score.txt"), ["PC PASS"])
    print(f"P1 {'PASS' if p1 else 'FAIL'}\n")

    print("P2 (a) KT1, four designs")
    p2 = same_numbers("designs", find("kt1_*/kt1.json"))
    p2 &= score_text("designs", find("kt1_*/score.txt"), ["KT1 MARGINAL"])
    print("P2 (b) KT2 and KT3, variants of the fix")
    p2 &= same_numbers("fix_variants", find("kt23_*/kt.json"))
    expect = [
        "KT3 primary (learning curve): KILL",
        "KT3 secondary (short window): KILL",
    ]
    p2 &= score_text("fix_variants", find("kt23_*/score.txt"), expect)
    p2 &= "-> KT2 MARGINAL" in (ROOT / "results" / "fix_variants_score.txt").read_text()
    print(f"P2 {'PASS' if p2 else 'FAIL'}")


if __name__ == "__main__":
    main()
