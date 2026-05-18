"""CLI default-value tests for ``benchmarks/scripts/run_benchmark.py``.

The CLI argparse defaults must match the config.py defaults that already
align with the CLAM README invocation (commits e446547..6e421c7). A
CLI-side regression that re-introduces ``--lr 1e-4`` or ``--n_folds 5``
would silently override the config and produce non-faithful runs while
all unit tests on TrainConfig still pass. This guards against that.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


def _load_run_benchmark_module():
    """Import benchmarks/scripts/run_benchmark.py without executing main()."""
    script_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts"
    )
    path = os.path.join(script_dir, "run_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_benchmark_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parser():
    mod = _load_run_benchmark_module()
    # parse_args() reads sys.argv; pass --dataset to satisfy required arg
    # without triggering further setup.
    monkey_argv = ["run_benchmark.py", "--dataset", "placeholder"]
    real_argv = sys.argv
    sys.argv = monkey_argv
    try:
        args = mod.parse_args()
    finally:
        sys.argv = real_argv
    return args


def test_default_lr_matches_clam_readme(parser):
    """CLAM README: ``--lr 2e-4``. CLI default must match."""
    assert parser.lr == 2e-4, (
        f"CLI --lr default {parser.lr} drifted from CLAM README's 2e-4. "
        "If lowering, update config.py TrainConfig.lr and the methods "
        "section together."
    )


def test_default_n_folds_matches_clam_readme(parser):
    """CLAM README: ``--k 10``. CLI default must match."""
    assert parser.n_folds == 10, (
        f"CLI --n_folds default {parser.n_folds} drifted from CLAM "
        "README's 10. Patient-stratified splits also assume the audit's "
        "10-fold projected-class-count math; reducing this number weakens "
        "minority-class coverage."
    )


def test_early_stopping_on_by_default(parser):
    """CLAM README: ``--early_stopping`` flag present. CLI inverts via
    ``--no_early_stopping``; default-off-of-no means on."""
    assert parser.no_early_stopping is False


def test_weighted_sample_on_by_default(parser):
    """CLAM README: ``--weighted_sample`` flag present."""
    assert parser.no_weighted_sample is False


def test_seed_and_max_epochs_match_config(parser):
    """Less critical but locked: any drift requires a deliberate update."""
    assert parser.seed == 42
    assert parser.max_epochs == 200
    assert parser.patience == 20
    assert parser.stop_epoch == 50
