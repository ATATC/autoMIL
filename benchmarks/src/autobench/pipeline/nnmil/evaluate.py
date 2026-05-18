"""Normalize nnMIL metrics to shared benchmark format.

Methods-note on AUC formula provenance
--------------------------------------
The CLAM and nnMIL wrapper paths get their AUC from different code paths,
but the two formulas are mathematically equivalent in the common case.

- CLAM path:  ``pipeline/evaluate.py::compute_extended_metrics`` recomputes
  AUC from predictions using upstream CLAM's per-class ``roc_curve`` +
  ``nanmean`` formula (``lib/CLAM/utils/core_utils.py:514-527``).
- nnMIL path: the AUC value passed in via ``raw_metrics["{split}/auroc"]``
  is produced by nnMIL's trainer using
  ``sklearn.metrics.roc_auc_score(multi_class='ovr', average='macro')``
  (``lib/nnMIL/utilities/utils.py:130-141``). We map it through without
  recomputation.

These two formulas compute the SAME thing for binary tasks and for
multi-class tasks where every class appears in every test fold. The
inner binary AUC per class is identical; the only difference is how
missing classes are handled in a fold:

- CLAM's ``nanmean`` skips a class with zero positives and averages
  over present classes.
- ``roc_auc_score(multi_class='ovr')`` raises ``ValueError`` in the same
  case (sklearn refuses an undefined macro-mean).

Each path matches its own upstream's published behaviour. Numbers
should agree to floating-point noise unless a minority class is
literally absent from some test fold, in which case CLAM's path
degrades gracefully while sklearn's would have crashed upstream too.
"""

from __future__ import annotations

# nnMIL's evaluate(split='test') returns keys like "test_test/bacc", "test_test/auroc", etc.
# We map these to our unified metric names used by compute_confidence_intervals().

_NNMIL_TO_SHARED: dict[str, str] = {
    "acc": "accuracy",
    "bacc": "balanced_accuracy",
    "auroc": "auc_roc",
    "weighted_f1": "f1",
    "kappa": "kappa",
}


def normalize_nnmil_metrics(raw_metrics: dict, split: str = "test") -> dict[str, float]:
    """Map nnMIL metric keys to the shared benchmark schema.

    nnMIL returns keys like ``{split}_{split}/bacc`` (e.g. ``test_test/bacc``).
    We extract the metric suffix and map to our standard names.

    Returns a dict compatible with ``compute_extended_metrics`` output
    (keys: auc_roc, accuracy, balanced_accuracy, f1, sensitivity, specificity).

    The ``auc_roc`` value here is the OvR-macro AUC produced by nnMIL's
    trainer; see the module docstring for the provenance asymmetry vs. the
    CLAM path.
    """
    result: dict[str, float] = {}

    for raw_key, value in raw_metrics.items():
        # Extract the metric suffix after the last "/"
        if "/" not in raw_key:
            continue
        suffix = raw_key.rsplit("/", 1)[1]
        if suffix in _NNMIL_TO_SHARED:
            result[_NNMIL_TO_SHARED[suffix]] = float(value)

    # nnMIL doesn't compute sensitivity/specificity; set to NaN
    result.setdefault("sensitivity", float("nan"))
    result.setdefault("specificity", float("nan"))

    return result
