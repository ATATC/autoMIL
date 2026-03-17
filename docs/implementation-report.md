# autoMIL Implementation Report

## Goal

Build a standalone, open-source framework that enables any coding agent to
autonomously improve ML models through persistent experimentation, knowledge
accumulation, and multi-branch exploration. The framework overlays onto
existing project repos, capturing full-codebase modifications per experiment
via git worktree snapshots.

autoMIL is the first sub-project of three toward a publication:
1. **Framework extraction** (this implementation) - decouple from ovarian cancer project
2. **Multi-cancer validation** - run on 3 cancer types (ovarian HRD, CLWD lung, TBD)
3. **Paper** - benchmarks vs Optuna, analysis, arXiv preprint

## Problem

MIL model development involves tedious manual iteration: try hyperparameters,
losses, architectures, evaluate, repeat. Existing AutoML tools (Optuna, etc.)
only search predefined parameter spaces. They cannot invent new architectures,
combine techniques creatively, or learn from failed experiments. Meanwhile,
coding agents can read code, design experiments, and modify any file, but they
lack infrastructure for persistent tracking, parallel execution, and knowledge
retention across sessions.

## Solution

autoMIL provides the orchestration layer between a coding agent and the
experiment execution environment:

```
Coding Agent (designs experiments)
    |
    | edits files, runs CLI
    v
automil CLI  -->  Experiment Graph (tracks tree of experiments)
    |
    | snapshots changed files
    v
Orchestrator  -->  GPU Scheduler (bin-packing, parallel execution)
    |
    | git worktree + overlay
    v
Isolated Execution  -->  result.json  -->  Graph Promotion
    |
    v
3D Visualization Dashboard (live SSE updates)
```

## Architecture

### Two-layer design

**Framework layer** (installed as Python package):
- `graph.py` - Directed tree tracking experiments with UCB-inspired scoring
- `runner.py` - Git worktree overlay for isolated parallel execution
- `orchestrator.py` - GPU scheduler daemon with best-fit bin packing
- `cli.py` - Click-based CLI wrapping all operations
- `viz/` - Real-time 3D dashboard (aiohttp + SSE + Three.js/ForceGraph3D)
- `templates/` - Jinja2 templates for project initialization

**Project layer** (created by `automil init` inside existing repos):
- `automil/config.yaml` - Project settings, editable/readonly file lists
- `automil/program.md` - Agent instructions for the experiment loop
- `automil/learnings.md` - Accumulated insights (what works, what doesn't)
- `automil/graph.json` - Experiment tree state (runtime, gitignored)
- `automil/orchestrator/` - Queue, archive, completed directories

### Key innovation: git worktree overlay

The agent can modify any file in the repo. When it submits an experiment:
1. Only the changed files are copied to `archive/node_NNNN/`
2. The orchestrator creates a lightweight git worktree at the base commit
3. Changed files are overlaid on top
4. The experiment runs in this isolated environment
5. The worktree is cleaned up after completion

This means each experiment stores only its diff (a few files), not the
entire repo, while still running in a complete project environment. Multiple
experiments run in parallel on different GPUs without file conflicts.

### Experiment graph

Experiments form a directed tree. Each node has a parent ("built upon")
edge. Nodes are scored using a hybrid UCB formula that balances exploitation
(build on best results) with exploration (try under-explored branches).
The agent uses `automil rank` to pick diverse experiments across branches.

Keep/discard is computed by the framework via Pareto dominance: a node
is "keep" only if it does not regress on any tracked metric compared to
its parent.

### Agent compatibility

Claude Code is the first-class experience (skill + stop-prevention hook).
Any coding agent that can read files, edit code, and run shell commands
works via the CLI + program.md interface. Documented for Cursor, Codex,
Aider, and Windsurf.

## What was built

### Package structure

```
autoMIL/
  src/automil/
    __init__.py          # v0.1.0
    graph.py             # 483 lines - experiment tree tracking
    runner.py            # 75 lines - git worktree overlay
    orchestrator.py      # 747 lines - GPU scheduler daemon
    cli.py               # ~350 lines - click-based CLI
    viz/
      server.py          # 264 lines - SSE + aiohttp
      static/
        index.html       # dashboard shell
        app.js           # 632 lines - 3D force graph
        style.css         # dark theme with glassmorphism
    templates/
      config.yaml.j2
      program.md.j2
      learnings.md.j2
      .gitignore.j2
  tests/
    test_graph.py        # 26 tests
    test_runner.py       # 7 tests
    test_cli.py          # 5 tests
    test_integration.py  # 7 tests (end-to-end)
  examples/
    ovarian_hrd/automil/ # 189-experiment graph, learnings
    clwd/automil/        # lung adenocarcinoma skeleton
    placeholder/automil/ # minimal template
  .claude/
    skills/automil.md    # Claude Code skill
    hooks/on_stop.sh     # stop prevention
  docs/
    getting-started.md
    agent-compatibility.md
  pyproject.toml
  README.md
  LICENSE                # Apache 2.0
```

### CLI commands

| Command | Description |
|---------|-------------|
| `automil init` | Add autoMIL to an existing git repo |
| `automil submit --node <id> --desc "..." --files <f>` | Snapshot changed files, queue experiment |
| `automil rank` | Show top-ranked proposals |
| `automil propose --parent <id> --desc "..."` | Add a brainstormed proposal |
| `automil reconcile` | Sync graph with orchestrator state |
| `automil status` | Show experiment summary |
| `automil start-loop` / `automil stop-loop` | Control agent loop flag |
| `automil orchestrator start/stop/status` | Manage GPU scheduler |
| `automil viz start/stop/status` | Manage 3D dashboard |

### Key changes from the ovarian-coupled version

| Aspect | Before (ovarian) | After (standalone) |
|--------|-------------------|-------------------|
| Paths | Hardcoded `/mnt/pool/...` | Config-driven, no internal paths |
| Init model | N/A (manual setup) | `automil init` overlays onto existing repo |
| Experiment submission | `script_inline` in JSON spec | Git-diff overlay (only changed files stored) |
| Execution | Direct subprocess with temp copy | Git worktree + file overlay |
| Result contract | Log parsing + results.tsv fallback | train.py writes `result.json` |
| Keep/discard | train.py computes from `tsv_status` | Framework computes via Pareto dominance |
| results.tsv | Written by train.py in worktree | Written by orchestrator from result.json |
| Technique tags | Hardcoded module-level dict | Configurable via constructor |
| Config hash | Single-file tokenizer hash | Multi-file manifest hash with base_commit |
| Agent coupling | Claude Code only | CLI + program.md for any agent |

## Test results

45 tests, all passing:
- 26 graph tests (node lifecycle, scoring, reconciliation, migration, multi-file hash)
- 7 runner tests (worktree create/cleanup, overlay, result collection)
- 5 CLI tests (init, submit, rank)
- 7 integration tests (end-to-end flow, path sanitization, multi-submit, propose+rank)

## Validation with existing data

The ovarian HRD example ships with a static `graph.json` containing 189
experiments and a `learnings.md` with accumulated insights from the original
autonomous loop. Best composite: 0.851 (from 0.814 baseline, +4.5%
improvement discovered autonomously).

## Next steps

1. **Dry run**: Initialize autoMIL in the ovarian repo, verify the full
   loop works end-to-end with the new overlay model
2. **Sub-project 2**: Run autoMIL on CLWD (lung adenocarcinoma, 408 WSIs,
   7-class subtyping) and a third dataset (TBD, supervisor decision)
3. **Sub-project 3**: Optuna comparison baselines, analysis notebooks,
   paper manuscript for arXiv
