# autoMIL

Autonomous agent-driven Multiple Instance Learning for computational pathology.

autoMIL overlays onto your existing ML project, enabling any coding agent to
autonomously explore architectures, hyperparameters, and training strategies.
The agent edits your codebase, the orchestrator runs experiments on GPUs in
parallel, and results feed back into the next iteration.

## Features

- **Repo overlay**: Adds to your existing project, the agent can modify any file
- **Agent-agnostic**: Works with Claude Code, Cursor, Codex, Aider, or any agent with file and shell access
- **Experiment graph**: Tree-based tracking with UCB-inspired scoring for multi-branch exploration
- **Git worktree isolation**: Each experiment runs in a snapshot, only changed files are stored
- **GPU orchestrator**: Background scheduler with best-fit bin packing across GPUs
- **3D visualization**: Interactive dashboard for exploring the experiment tree
- **Persistent knowledge**: Learnings accumulate across sessions via learnings.md

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
cd /path/to/your/project    # an existing git repo
automil init                 # adds automil/ subdirectory
# edit automil/config.yaml with your project settings
automil orchestrator start   # start GPU scheduler
# launch your coding agent (e.g., claude, cursor)
```

## Documentation

- [Getting Started](docs/getting-started.md) - Setup, configuration, and usage
- [Agent Compatibility](docs/agent-compatibility.md) - Guides for different coding agents
- [Implementation Report](docs/implementation-report.md) - Architecture and design decisions

## Examples

See `examples/` for reference configurations:
- `ovarian_hrd/` - Binary classification with 189 autonomous experiments
- `clwd/` - Multi-class lung adenocarcinoma subtyping
- `placeholder/` - Minimal template

## License

Apache 2.0. See [LICENSE](LICENSE).
