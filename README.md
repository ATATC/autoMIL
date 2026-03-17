# autoMIL

Autonomous agent-driven Multiple Instance Learning for computational pathology.

autoMIL pairs a coding agent with a GPU scheduler to autonomously explore MIL architectures, hyperparameters, and training strategies. The agent proposes experiments, the orchestrator runs them, and results feed back into the next iteration.

## Features

- **Agent-agnostic**: Works with Claude Code, Cursor, Codex, Aider, Windsurf, or any agent with file and shell access
- **Experiment graph**: DAG-based tracking of experiment lineage, proposals, and results
- **GPU orchestrator**: Background scheduler that queues and runs training jobs
- **3D visualization**: Interactive dashboard for exploring the experiment landscape
- **CLI-driven**: All operations available via the `automil` command

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
automil init my_project
cd my_project
automil orchestrator start
# Start your coding agent and run the experiment loop
```

## Documentation

- [Getting Started](docs/getting-started.md) - Full setup, configuration, and first run
- [Agent Compatibility](docs/agent-compatibility.md) - Setup guides for different coding agents

## Examples

See `examples/` for complete worked examples including ovarian cancer HRD classification and lung adenocarcinoma subtyping.

## License

See [LICENSE](LICENSE) for details.
