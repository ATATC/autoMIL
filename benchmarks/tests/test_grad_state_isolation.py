"""Regression tests for process-global torch-state isolation between experiments.

The single-GPU ``run_benchmark`` path executes heterogeneous frameworks
back-to-back in **one process / one thread**.  Several torch switches are
thread-local *globals*, so a trainer that returns with one of them changed can
poison the next experiment:

* ``torch.set_grad_enabled`` — nnMIL's ``ClassificationTrainer`` used to leave
  it ``False`` (a bare ``torch.set_grad_enabled(False)`` with no restore), so
  the *next* CLAM experiment's first ``total_loss.backward()`` raised::

      RuntimeError: element 0 of tensors does not require grad and does not
      have a grad_fn

* ``torch.use_deterministic_algorithms`` — nnMIL enables it globally and CLAM
  never sets it, so a CLAM experiment's kernels (and thus metrics) would depend
  on whether an nnMIL run preceded it — an unfair, order-dependent result.

These tests pin the fix: ``orchestrator._isolated_torch_state`` restores the
pristine global state (grad on, deterministic-algorithms off) around every
experiment dispatch, so no framework's global side effects leak forward.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline.config import Framework
from autobench.pipeline.orchestrator import (
    _isolated_torch_state,
    _run_single_experiment_dispatch,
)


def _one_train_step() -> None:
    """A minimal grad-requiring training step.

    Raises ``RuntimeError`` (the exact failure seen in the bug report) if the
    global autograd flag is disabled when the forward pass runs.
    """
    model = torch.nn.Linear(4, 1)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()


class TestGradStateLeak:
    def test_leak_reproduces_without_isolation(self) -> None:
        """Characterize the bug: a leaked disabled-grad flag breaks backward()."""
        prev = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)  # simulate a prior nnMIL experiment
            with pytest.raises(RuntimeError, match="does not require grad"):
                _one_train_step()
        finally:
            torch.set_grad_enabled(prev)

    def test_isolation_heals_incoming_leak(self) -> None:
        """The guard re-enables grad on entry even if a prior run left it off."""
        prev = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            with _isolated_torch_state():
                assert torch.is_grad_enabled()
                _one_train_step()  # must not raise
        finally:
            torch.set_grad_enabled(prev)

    def test_isolation_heals_outgoing_leak(self) -> None:
        """The guard leaves grad enabled on exit even if the body disabled it."""
        prev = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(True)
            with _isolated_torch_state():
                torch.set_grad_enabled(False)  # simulate a leaky framework
            assert torch.is_grad_enabled()
        finally:
            torch.set_grad_enabled(prev)

    def test_isolation_restores_on_exception(self) -> None:
        """Even if the experiment raises, grad must be left enabled."""
        prev = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            with pytest.raises(ValueError):
                with _isolated_torch_state():
                    torch.set_grad_enabled(False)
                    raise ValueError("boom")
            assert torch.is_grad_enabled()
        finally:
            torch.set_grad_enabled(prev)

    def test_isolation_resets_deterministic_algorithms(self) -> None:
        """The guard restores deterministic-algorithms to the off default.

        nnMIL flips this global on; CLAM never sets it.  Without the reset a
        CLAM experiment running after an nnMIL one would pick different kernels
        than one running first — an order-dependent (unfair) result.
        """
        prev_grad = torch.is_grad_enabled()
        prev_det = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # simulate nnMIL
            with _isolated_torch_state():
                assert torch.are_deterministic_algorithms_enabled() is False
            assert torch.are_deterministic_algorithms_enabled() is False  # reset on exit
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=True)
            torch.set_grad_enabled(prev_grad)


class TestDispatchIsolation:
    def test_dispatch_runs_with_grad_enabled_despite_prior_leak(self) -> None:
        """Each experiment dispatch starts grad-enabled and heals its own leak."""
        observed: dict[str, bool] = {}

        def fake_run_experiment(exp_cfg, benchmark_dir, device, wandb_project):
            observed["grad_on_entry"] = torch.is_grad_enabled()
            torch.set_grad_enabled(False)  # simulate a leaky framework
            return {"ok": True}

        exp_cfg = type("Exp", (), {"framework": Framework.CLAM})()

        prev = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)  # prior experiment left grad disabled
            with patch(
                "autobench.pipeline.clam.runner.run_experiment",
                side_effect=fake_run_experiment,
            ):
                _run_single_experiment_dispatch(
                    exp_cfg, "/tmp/bench", torch.device("cpu"), None,
                )
            assert observed["grad_on_entry"] is True
            assert torch.is_grad_enabled() is True  # leak healed on exit
        finally:
            torch.set_grad_enabled(prev)
