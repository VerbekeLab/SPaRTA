"""Loss factory: one reusable place for every training round."""

import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss   # already a project dependency (no custom code)


def make_loss(kind="bce", pos_weight=None, alpha=0.75, gamma=2.0):
    """Return a (logits, targets) -> scalar loss for binary classification.

      'bce'   : BCEWithLogitsLoss(pos_weight)  -- default; pos_weight balances the classes.
      'focal' : torchvision sigmoid focal loss -- alpha up-weights the rare POSITIVE class
                (alpha=0.75), gamma=2 down-weights easy examples. alpha REPLACES pos_weight,
                so pos_weight is intentionally ignored for focal (no double-counting).
    logits and targets are both shape (B,) float; the callable works unchanged for any model.
    """
    if kind == "bce":
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return lambda logits, targets: bce(logits, targets)
    if kind == "focal":
        return lambda logits, targets: sigmoid_focal_loss(
            logits, targets, alpha=alpha, gamma=gamma, reduction="mean")
    raise ValueError(f"unknown loss kind: {kind!r}")
