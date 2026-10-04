"""Exact small-sample arithmetic using only the standard library."""
import argparse
import json
import math
from common import index_rows, probability, read_jsonl


def counts(k, n):
    if type(k) is not int or type(n) is not int or n < 1 or not 0 <= k <= n:
        raise ValueError("require integer 0 <= k <= n and n > 0")


def binomial_cdf(k, n, p):
    if p == 0:
        return 1.0
    if p == 1:
        return float(k >= n)
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))


def _invert_cdf(k, n, target):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    counts(k, n)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    lower = 0.0 if k == 0 else _invert_cdf(k - 1, n, 1 - alpha / 2)
    upper = 1.0 if k == n else _invert_cdf(k, n, alpha / 2)
    return lower, upper


def zero_upper(n, alpha=0.05):
    counts(0, n)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return -math.expm1(math.log(alpha) / n)


def sample_size(bound, alpha=0.05):
    if not 0 < bound < 1 or not 0 < alpha < 1:
        raise ValueError("bound and alpha must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log1p(-bound))


def mcnemar(b, c):
    if type(b) is not int or type(c) is not int or b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative integers")
    return 1.0 if b + c == 0 else min(1.0, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))


def paired(left, right, threshold=0.70):
    a, b = index_rows(left), index_rows(right)
    if not a or set(a) != set(b):
        raise ValueError("paired files must contain the same non-empty id set")
    flips = 0; diffs = []
    for key in a:
        pa, pb = probability(a[key]["p_tpl"]), probability(b[key]["p_tpl"])
        flips += (pa >= threshold) != (pb >= threshold)
        diffs.append(abs(pa - pb))
    return {"paired_n": len(a), "decision_flips": flips, "max_abs_probability_change": max(diffs)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired", nargs=2, metavar=("ORIGINAL", "RENAMED"))
    a = p.parse_args()
    if a.paired:
        print(json.dumps(paired(*[read_jsonl(path) for path in a.paired]), indent=2)); return
    for k, n in ((4, 7), (6, 7), (4, 6), (2, 6), (16, 48), (19, 20), (29, 37), (6, 9), (5, 9)):
        lo, hi = clopper_pearson(k, n)
        print(f"{k}/{n}: {lo:.3f} to {hi:.3f}")
    for n in (41, 8, 48):
        print(f"0/{n}: one-sided upper {zero_upper(n):.6f}")
    print("zero-event sample sizes:", sample_size(0.01), sample_size(0.001))
    print(f"6/6: one-sided lower {0.05 ** (1 / 6):.3f}")
    print(f"McNemar 3 vs 1: p = {mcnemar(3, 1):.3f}")


if __name__ == "__main__":
    main()
