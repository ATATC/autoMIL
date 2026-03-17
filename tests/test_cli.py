"""Tests for the automil CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestInit:
    def test_creates_project(self, cli_runner, tmp_path):
        """automil init creates a complete project scaffold."""
        result = cli_runner.invoke(main, ["init", str(tmp_path / "myproject")])
        assert result.exit_code == 0

        proj = tmp_path / "myproject"
        assert (proj / "config.yaml").exists()
        assert (proj / "train.py").exists()
        assert (proj / "prepare.py").exists()
        assert (proj / "program.md").exists()
        assert (proj / ".gitignore").exists()
        assert (proj / "learnings.md").exists()
        assert (proj / "orchestrator" / "queue").is_dir()
        assert (proj / "orchestrator" / "archive").is_dir()
        assert (proj / "orchestrator" / "completed").is_dir()

    def test_creates_git_repo(self, cli_runner, tmp_path):
        """automil init initializes a git repo with initial commit."""
        proj = tmp_path / "myproject"
        cli_runner.invoke(main, ["init", str(proj)])

        # Check it's a git repo
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=proj, capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_gitignore_excludes_runtime(self, cli_runner, tmp_path):
        """Runtime files (graph.json, results.tsv, orchestrator/) are gitignored."""
        proj = tmp_path / "myproject"
        cli_runner.invoke(main, ["init", str(proj)])

        gitignore = (proj / ".gitignore").read_text()
        assert "graph.json" in gitignore
        assert "results.tsv" in gitignore
        assert "orchestrator/" in gitignore


class TestSubmit:
    def test_submit_captures_files(self, cli_runner, tmp_path, monkeypatch):
        """automil submit snapshots specified files to archive."""
        # Create a project
        proj = tmp_path / "myproject"
        cli_runner.invoke(main, ["init", str(proj)])

        # Modify train.py
        (proj / "train.py").write_text("print('modified')\n")

        # Change cwd so _find_project_root() works
        monkeypatch.chdir(proj)

        # Submit
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "test", "--files", "train.py"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # Check archive
        archive = proj / "orchestrator" / "archive" / "node_0001"
        assert (archive / "train.py").exists()
        assert (archive / "train.py").read_text() == "print('modified')\n"

        # Check spec in queue
        queue_files = list((proj / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        assert spec["id"] == "node_0001"
        assert "base_commit" in spec


class TestRank:
    def test_rank_outputs_proposals(self, cli_runner, tmp_path, monkeypatch):
        """automil rank shows top proposals from graph.json."""
        proj = tmp_path / "myproject"
        cli_runner.invoke(main, ["init", str(proj)])

        # Change cwd so _find_project_root() works
        monkeypatch.chdir(proj)

        # Create a graph with proposals (use correct API signatures)
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(proj / "graph.json"))
        root = graph.add_executed(
            parent_id=None,
            description="baseline",
            techniques=["baseline"],
            metrics={"test_auc": 0.85, "test_bacc": 0.80, "composite": 0.825},
            status="keep",
        )
        graph.add_proposed(
            parent_id=root,
            description="try focal loss",
            techniques=["focal"],
        )
        graph.save()

        result = cli_runner.invoke(main, ["rank"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "focal" in result.output
