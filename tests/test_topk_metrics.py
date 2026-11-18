# Checks for the top-k metrics in src/methods/evaluation.py: precision@k, recall@k, F1@k. Plain asserts: runs
# under pytest or directly, from anywhere (it locates the repo root itself) with the SPaRTA_env interpreter:
#   /Users/brunodeprez/anaconda3/envs/SPaRTA_env/bin/python tests/test_topk_metrics.py
#
# What is verified
#   1. by hand      small rankings with known answers: count k, fraction k (rounded up), the tie rule.
#   2. identities   R@k = P@k * k / positives; F1@k = 2PR/(P+R) = 2 hits/(k+positives); P@R = R@R = F1@R.
#   3. ceilings     a perfect ranking scores min(1, pos/k), min(1, k/pos) and 2 min(k, pos)/(k + pos).
#   4. bounds       F1@k never exceeds best_f1 (the best F1 over every threshold), even with tied scores.
#   5. edge cases   k above the number of rows gives NaN; no positives gives NaN recall / F1 (precision stays 0).
#   6. extra_metrics emits P@/R@/F1@ per k and keeps the F1@0.5 / F1_best* columns.
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from src.methods import evaluation as ev


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_by_hand():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])            # ranking: 1 0 1 0 0 1 -> hits 1, 1, 2, 2, 2, 3
    for k, hits in [(1, 1), (2, 1), (3, 2), (4, 2), (5, 2), (6, 3)]:
        p, r, f = ev.topk_metrics(y, s, k)
        assert close(p, hits / k) and close(r, hits / 3) and close(f, 2 * hits / (k + 3)), (k, p, r, f)
        assert close(ev.precision_at_k(y, s, k), p) and close(ev.recall_at_k(y, s, k), r) and close(ev.f1_at_k(y, s, k), f)
    # a fraction of the rows is rounded up to a count: 0.4 * 6 = 2.4 -> 3 rows
    assert close(ev.precision_at_k(y, s, 0.4), 2 / 3) and close(ev.recall_at_k(y, s, 0.4), 2 / 3)
    assert close(ev.f1_at_k(y, s, 0.4), 2 * 2 / (3 + 3))


def test_ties_are_resolved_in_expectation_not_by_row_order():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])                       # all tied: the one positive fills a top-2 slot with prob 1/2
    for order in (np.arange(4), np.arange(4)[::-1]):
        yy, ss = y[order], s[order]
        p, r, f = ev.topk_metrics(yy, ss, 2)
        assert close(p, 0.25) and close(r, 0.5) and close(f, 2 * 0.5 / (2 + 1))
    # a partial tie at the cut-off: one clear positive on top, then three tied rows holding one positive
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    assert close(ev.recall_at_k(y, s, 2), (1 + 1 / 3) / 2)     # 1 sure hit + 1 slot filled by a tied row (mean label 1/3)


def test_identities_on_random_scores_with_ties():
    rng = np.random.default_rng(1997)
    for _ in range(40):
        n = int(rng.integers(30, 300))
        y = (rng.random(n) < rng.uniform(0.02, 0.4)).astype(int)
        if y.sum() == 0:
            y[0] = 1
        s = np.round(rng.random(n) + 0.8 * y * rng.random(n), 1)          # coarse scores -> many ties
        pos = int(y.sum())
        for k in (1, 5, 25, n // 2, n, 0.1, 0.5):
            p, r, f = ev.topk_metrics(y, s, k)
            kk = ev._resolve_k(k, n)
            assert close(r, p * kk / pos), (k, p, r)                       # R@k = P@k * k / pos
            assert close(f, 2 * p * kk / (kk + pos)), (k, p, f)            # F1@k = 2 hits / (k + pos)
            assert p == 0 or close(f, 2 * p * r / (p + r)), (k, p, r, f)   # harmonic mean of P@k and R@k
        # k = number of positives: precision = recall = F1
        pr, rr, fr = ev.topk_metrics(y, s, pos)
        assert close(pr, rr) and close(rr, fr)
        assert close(ev.extra_metrics(y, s, r_precision=True)["P@R"], pr)


def test_perfect_ranking_hits_the_ceilings():
    for pos, n in [(5, 40), (60, 400), (100, 500)]:
        y = np.zeros(n, dtype=int)
        y[:pos] = 1
        s = np.linspace(1, 0, n)                                           # every positive ranked above every negative
        for k in (10, 50, 100, 500):
            if k > n:
                continue
            p, r, f = ev.topk_metrics(y, s, k)
            assert close(p, min(1, pos / k)) and close(r, min(1, k / pos)) and close(f, 2 * min(k, pos) / (k + pos)), (pos, k)


def test_f1_at_k_never_exceeds_best_f1():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(30, 300))
        y = (rng.random(n) < 0.15).astype(int)
        if y.sum() == 0:
            y[0] = 1
        s = np.round(rng.random(n) + 0.7 * y * rng.random(n), int(rng.integers(1, 3)))   # ties of varying coarseness
        top = ev.best_f1(y, s)["F1"]
        for k in (1, 3, int(y.sum()), n // 4, n // 2, n):
            assert ev.f1_at_k(y, s, k) <= top + 1e-12, (k, ev.f1_at_k(y, s, k), top)


def test_edge_cases():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    for k in (5, 100):                                                     # more rows than exist
        assert all(np.isnan(v) for v in ev.topk_metrics(y, s, k))
    assert all(np.isnan(v) for v in ev.topk_metrics(y, s, 0))              # k below 1
    p, r, f = ev.topk_metrics(np.zeros(4, dtype=int), s, 2)                # no positives: precision is a real 0
    assert p == 0 and np.isnan(r) and np.isnan(f)


def test_extra_metrics_columns():
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.2).astype(int)
    s = rng.random(200) + 0.5 * y
    out = ev.extra_metrics(y, s, ks=(10, 50), fracs=(0.1,), r_precision=True)
    old = ["F1@0.5", "F1_best", "F1_best_threshold", "F1_best_precision", "F1_best_recall", "P@10", "P@50", "P@10%", "P@R"]
    new = ["R@10", "R@50", "R@10%", "F1@10", "F1@50", "F1@10%"]
    assert all(c in out for c in old + new), [c for c in old + new if c not in out]
    assert "R@R" not in out and "F1@R" not in out                          # equal to P@R by definition
    assert close(out["F1@10"], ev.f1_at_k(y, s, 10)) and close(out["R@10%"], ev.recall_at_k(y, s, 0.1))
    default = ev.extra_metrics(y, s)                                       # default k = 100 only
    assert {"P@100", "R@100", "F1@100"} <= set(default) and "P@500" not in default
    assert close(default["F1@0.5"], ev.f1_at_threshold(y, s, 0.5))         # the score cut-off is untouched


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
