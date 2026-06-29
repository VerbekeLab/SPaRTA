import matplotlib.pyplot as plt
import numpy as np
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
