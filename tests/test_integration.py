"""End-to-end integration tests."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


def _init_git_repo(path: Path):
    """Initialize a git repo with an initial commit at the given path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, capture_output=True,
    )
    # Create a dummy file and commit so HEAD exists
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path, capture_output=True, check=True,
    )


class TestEndToEnd:
    def test_init_submit_flow(self, tmp_path, monkeypatch):
        """Full flow: init in existing repo, submit experiment, check archive."""
        runner = CliRunner()

        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Init autoMIL overlay
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        # Verify automil subdirectory
        adir = tmp_path / "automil"
        assert (adir / "config.yaml").exists()
        assert (adir / "program.md").exists()
        assert (adir / "learnings.md").exists()
        assert (adir / ".gitignore").exists()

        # No train.py or prepare.py
        assert not (adir / "train.py").exists()
        assert not (adir / "prepare.py").exists()
        assert not (tmp_path / "train.py").exists()
        assert not (tmp_path / "prepare.py").exists()

        # Create a file and submit
        (tmp_path / "model.py").write_text("print('experiment 1')\n")
        result = runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "test exp",
             "--files", "model.py"],
        )
        assert result.exit_code == 0, result.output

        # Verify archive structure
        archive = adir / "orchestrator" / "archive" / "node_0001"
        assert (archive / "model.py").exists()
        assert (archive / "model.py").read_text() == "print('experiment 1')\n"

        # Verify queue spec
        queue_files = list((adir / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        assert spec["id"] == "node_0001"
        assert "base_commit" in spec
        assert "overlay_manifest" in spec

        # Status should work
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0, result.output

    def test_no_internal_paths_in_package(self):
        """Verify no institution-specific paths leaked into the package."""
        import automil
        pkg_dir = Path(automil.__file__).parent

        for f in pkg_dir.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".md", ".yaml", ".j2", ".html", ".js"):
                content = f.read_text()
                assert "/mnt/pool" not in content, f"Internal path in {f}"
                assert "/home/jma" not in content, f"Home path in {f}"

    def test_multiple_submits(self, tmp_path, monkeypatch):
        """Multiple experiments can be submitted sequentially."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])

        for i in range(3):
            (tmp_path / "model.py").write_text(f"print('experiment {i}')\n")
            result = runner.invoke(
                main,
                ["submit", "--node", f"node_{i:04d}", "--desc", f"exp {i}",
                 "--files", "model.py"],
            )
            assert result.exit_code == 0, result.output

        # All 3 should be in queue
        queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 3

        # All 3 archives should exist with different content
        for i in range(3):
            archive = tmp_path / "automil" / "orchestrator" / "archive" / f"node_{i:04d}"
            assert (archive / "model.py").read_text() == f"print('experiment {i}')\n"

    def test_propose_and_rank(self, tmp_path, monkeypatch):
        """Propose experiments and rank them."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])

        # Create a graph with a root node
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None,
            description="baseline",
            techniques=["baseline"],
            metrics={"test_auc": 0.85, "test_bacc": 0.80, "composite": 0.825},
            status="keep",
        )
        graph.save()

        # Propose via CLI
        result = runner.invoke(
            main,
            ["propose", "--parent", root, "--desc", "try focal loss",
             "--techniques", "focal"],
        )
        assert result.exit_code == 0, result.output

        # Rank should show the proposal
        result = runner.invoke(main, ["rank"])
        assert result.exit_code == 0, result.output
        assert "focal" in result.output

    def test_start_stop_loop(self, tmp_path, monkeypatch):
        """start-loop and stop-loop manage the flag file."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])

        result = runner.invoke(main, ["start-loop"])
        assert result.exit_code == 0
        assert (tmp_path / ".automil_active").exists()

        result = runner.invoke(main, ["stop-loop"])
        assert result.exit_code == 0
        assert not (tmp_path / ".automil_active").exists()
