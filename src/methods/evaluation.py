import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve, auc)

from src.utils.visualisation import OKABE_ITO


def evaluate_scores(y_true, scores):
    """Return (AUC-PR, AUC-ROC). AUC-PR is the primary metric."""
    return average_precision_score(y_true, scores), roc_auc_score(y_true, scores)


def plot_curves(scores_dict, y_true, title, save_path=None):
    """One ROC (left) + one PR (right) axes comparing every model in
    `scores_dict` (name -> np.ndarray of test probabilities) on the same split.
    Colours from the Okabe-Ito palette by index. ROC carries the chance diagonal,
    PR the positive-prevalence no-skill line; legends report AUC / AP.
    Saves to `save_path` (headless-safe) if given, else shows the figure."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, s) in enumerate(scores_dict.items()):
        c = OKABE_ITO[i % len(OKABE_ITO)]
        fpr, tpr, _ = roc_curve(y_true, s)
        ax_roc.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax_pr.plot(rec, prec, color=c, lw=2, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    # Reference lines: ROC chance diagonal, PR positive-prevalence (no-skill) line.
    ax_roc.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.axhline(y_true.mean(), ls="--", color="grey", lw=1, label=f"prevalence ({y_true.mean():.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve")
    ax_pr.legend(loc="upper right", fontsize=8)
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.suptitle(title); fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path); plt.close(fig)
    else:
        plt.show()


def write_metrics(path, scores_dict, y_true):
    """Per-model AUC-PR / AUC-ROC plus the ROC (fpr, tpr) and PR (precision,
    recall) arrays, in the style of scripts/experiment_features.py result files."""
    with open(path, 'w') as f:
        for name, s in scores_dict.items():
            aucpr, aucroc = evaluate_scores(y_true, s)
            fpr, tpr, _ = roc_curve(y_true, s)
            precision, recall, _ = precision_recall_curve(y_true, s)
            f.write(f'Model: {name}\n')
            f.write(f'AUC PR: {aucpr}\n')
            f.write(f'AUC ROC: {aucroc}\n')
            f.write(f'FPR: {fpr}\n')
            f.write(f'TPR: {tpr}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write('\n')


# --- Persisted test-band predictions ---------------------------------------------------------
# write_metrics keeps only AUC-PR / AUC-ROC plus the curve arrays, and those are written with
# str(ndarray), which numpy abbreviates with '...' beyond 1000 elements -- so the .txt files can
# NOT be used to rebuild the curves or to compute any other metric. The per-sample test scores
# below can: any threshold-free or top-k metric (precision@k, recall@k, lift, F1 at a
# threshold, calibration, bootstrap CIs, McNemar/DeLong between models) is then an offline
# calculation on one parquet instead of a retrain.

PRED_DIR = "results/predictions"


def save_predictions(path, scores_dict, y_true, anchors=None, nodes=None):
    """One parquet per metrics dump: y_true, optional anchor/node ids, one `score_<model>`
    column per model (float64, the same values write_metrics scored). Rows are the test
    band in the order the models were scored in, so models share rows and can be paired.
    Written to a temp file and os.replace()d, like feature_io.save_table."""
    df = pd.DataFrame({"y_true": np.asarray(y_true).astype(np.int8)})
    if anchors is not None:
        df["anchor"] = np.asarray(anchors).astype(np.int32)
    if nodes is not None:
        nodes = np.asarray(nodes)
        df["node"] = nodes.astype(str) if nodes.dtype == object else nodes
    for name, s in scores_dict.items():
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if len(s) != len(df):
            raise ValueError(f"{name}: {len(s)} scores for {len(df)} test rows")
        df[f"score_{name}"] = s
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    return path


def load_predictions(path):
    """(y_true, {model: scores}, df) from a save_predictions parquet."""
    df = pd.read_parquet(path)
    scores = {c[len("score_"):]: df[c].to_numpy() for c in df.columns if c.startswith("score_")}
    return df["y_true"].to_numpy(), scores, df


_AUC_BLOCK = re.compile(r"AUC PR:\s*([0-9.eE+\-]+)\s*\n\s*AUC ROC:\s*([0-9.eE+\-]+)")
_AUC_BLOCK_ROC_FIRST = re.compile(r"AUC ROC:\s*([0-9.eE+\-]+)\s*\n\s*AUC PR:\s*([0-9.eE+\-]+)")


def stored_metrics(txt_glob):
    """{model: (AUC_PR, AUC_ROC)} read from every metrics dump matching `txt_glob`.
    Handles write_metrics blocks ('Model: x'); the single-model static dumps have no 'Model:'
    line (and write AUC ROC first), so they are keyed by their file path instead."""
    out = {}
    for path in sorted(glob.glob(txt_glob)):
        text = open(path).read()
        blocks = re.split(r"^Model:\s*", text, flags=re.MULTILINE)
        if len(blocks) == 1:
            m = _AUC_BLOCK.search(text)
            if m:
                out[path] = (float(m.group(1)), float(m.group(2)))
            else:
                m = _AUC_BLOCK_ROC_FIRST.search(text)
                if m:
                    out[path] = (float(m.group(2)), float(m.group(1)))
            continue
        for b in blocks[1:]:
            name = b.split("\n", 1)[0].strip()
            m = _AUC_BLOCK.search(b)
            if m:
                out[name] = (float(m.group(1)), float(m.group(2)))
    return out


def compare_to_metrics(txt_glob, scores_dict, y_true, tol=1e-4):
    """Print reproduced vs on-disk AUC-PR / AUC-ROC for every model in `scores_dict`, flagging
    DIFF when either moves by more than `tol`. Sequence models reloaded from .pt should match to
    float noise; refit tabular baselines can drift slightly (multithreaded XGBoost, GPU MLP).
    A model with no stored block is reported as such. Returns {model: True/False/None}."""
    stored = stored_metrics(txt_glob)
    ok = {}
    for name, s in scores_dict.items():
        pr, roc = evaluate_scores(y_true, s)
        ref = stored.get(name)
        if ref is None:   # single-model static dumps are keyed by path: match on the file name
            ref = next((v for k, v in stored.items() if k.endswith(f"_{name}.txt")), None)
        if ref is None:
            print(f"[rescore] {name}: AUC-PR {pr:.6f} AUC-ROC {roc:.6f} (no stored metrics to compare)")
            ok[name] = None
            continue
        d_pr, d_roc = pr - ref[0], roc - ref[1]
        good = abs(d_pr) <= tol and abs(d_roc) <= tol
        ok[name] = good
        print(f"[rescore] {name}: AUC-PR {pr:.6f} (stored {ref[0]:.6f}, d={d_pr:+.2e}) "
              f"AUC-ROC {roc:.6f} (stored {ref[1]:.6f}, d={d_roc:+.2e}) -> {'MATCH' if good else 'DIFF'}")
    return ok


# --- Threshold / top-k metrics (computed from saved predictions, never inside a training job) --
# AUC-PR stays the primary metric. These two views complement it: precision@k is the analyst-
# workload view (of the k accounts a model flags first, how many are laundering), and F1 is the
# single-operating-point view.

def precision_at_k(y_true, scores, k):
    """Precision among the k highest-scored rows. `k` is a count (int >= 1) or a fraction of the
    test rows (float in (0, 1), rounded up). Ties at the cut-off score are resolved in
    EXPECTATION (the tied rows contribute their mean label for the slots they fill) instead of by
    row order: probabilities saturate at 0/1 often enough that row order, which is just anchor
    order, would otherwise decide the number. Returns NaN when k exceeds the number of rows."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    k = int(np.ceil(k * n)) if isinstance(k, float) and 0 < k < 1 else int(k)
    if k < 1 or k > n:
        return float("nan")
    cutoff = np.partition(s, n - k)[n - k]          # k-th largest score
    above = s > cutoff
    tied = s == cutoff
    slots = k - int(above.sum())                    # how many cut-off-tied rows make the top k
    return float((y[above].sum() + slots * y[tied].mean()) / k)


def best_f1(y_true, scores):
    """Maximum F1 over all score thresholds, with the threshold and the precision/recall there.
    This is an ORACLE operating point: the threshold is picked on the same rows it is scored
    on, so it is optimistic and is NOT a deployable F1. It is comparable across models (each at
    its own best cut-off, which also sidesteps uncalibrated scores such as IsolationForest's or
    pos_weight-shifted probabilities), but it is not what a threshold fixed in advance would
    give. Returns {'F1', 'threshold', 'precision', 'recall'}."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    p, r = precision[:-1], recall[:-1]               # the final (1, 0) point has no threshold
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return {"F1": float(f1[i]), "threshold": float(thresholds[i]),
            "precision": float(p[i]), "recall": float(r[i])}


def f1_at_threshold(y_true, scores, threshold=0.5):
    """F1 of the hard rule `score >= threshold` (predict laundering). A FIXED cut-off, so unlike
    best_f1 it is not tuned on the test rows -- but it is only meaningful for scores that are
    probabilities on a common scale. The sigmoid outputs of the supervised models are, though
    `pos_weight` shifts them upward; IsolationForest's anomaly scores (-score_samples, roughly
    0.3-0.8) are not, so 0.5 is arbitrary for them. F1 is 0 when nothing is predicted positive."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores, dtype=float) >= threshold
    tp = int((pred & y).sum())
    denom = int(pred.sum()) + int(y.sum())        # 2tp + fp + fn = predicted positives + actual positives
    return 2 * tp / denom if denom else 0.0


def _k_label(k):
    return f"P@{k * 100:g}%" if isinstance(k, float) and 0 < k < 1 else f"P@{int(k)}"


def extra_metrics(y_true, scores, ks=(100,), fracs=(), r_precision=False):
    """One flat dict per model: F1 at the fixed 0.5 cut-off ('F1@0.5'), best-F1 (oracle, see
    best_f1), plus precision@k for each count in `ks`, each fraction in `fracs` (e.g. 'P@1%'),
    and, if `r_precision`, P@R (k = number of positives). Defaults: P@100 only. Counts above
    the number of test rows come back NaN."""
    y = np.asarray(y_true)
    bf = best_f1(y, scores)
    out = {"F1@0.5": f1_at_threshold(y, scores, 0.5),
           "F1_best": bf["F1"], "F1_best_threshold": bf["threshold"],
           "F1_best_precision": bf["precision"], "F1_best_recall": bf["recall"]}
    for k in list(ks) + [float(f) for f in fracs]:
        out[_k_label(k)] = precision_at_k(y, scores, k)
    if r_precision:
        out["P@R"] = precision_at_k(y, scores, int(y.sum()))
    return out
