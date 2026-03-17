# Getting Started with autoMIL

## Prerequisites

- Python 3.10+
- NVIDIA GPU(s) with CUDA
- Git
- A coding agent (Claude Code recommended, or Cursor/Codex/Aider)

## Installation

```bash
git clone https://github.com/your-org/autoMIL.git
cd autoMIL
pip install -e .
# Or with ML dependencies:
pip install -e ".[ml]"
```

## Creating a Project

```bash
automil init my_project
cd my_project
```

This creates a git-initialized project with:
- `config.yaml` - Project configuration (paths, task, encoders, training params)
- `train.py` - Training script (agent-editable)
- `prepare.py` - Data loading utilities (typically read-only)
- `program.md` - Agent instructions for the experiment loop
- `learnings.md` - Accumulated insights
- `orchestrator/` - Runtime directories for experiment management

## Configuration

Edit `config.yaml` with your project-specific settings:

1. **Data paths**: Set `data.features_dir`, `data.splits_dir`, `data.mapping_csv`
2. **Task**: Set `task.name`, `task.type` (binary/multiclass), `task.label_column`
3. **Encoders**: List available encoders with their dimensions
4. **Baseline**: Set your starting performance numbers

## Implementing Data Loading

Edit `prepare.py` to implement `create_fold_loaders()` for your dataset. This function must return `(train_loader, val_loader, test_loader)` for each fold.

## Setting Up the Model

Edit `train.py` to implement:
- `create_model()` - Your MIL architecture
- `train_single_fold()` - The training loop for one fold

The training script must write `result.json` at completion (the template handles this).

## Running the Loop

### 1. Start the orchestrator
```bash
automil orchestrator start
```

### 2. Start the visualization dashboard (optional)
```bash
automil viz start
# Open http://localhost:8420 in your browser
```

### 3. Launch your coding agent

**Claude Code:**
```bash
claude
# Then type: /automil
```

**Other agents:** Point them at `program.md` and tell them to follow the instructions.

### 4. Monitor progress
```bash
automil status          # Quick summary
automil viz start       # 3D dashboard at localhost:8420
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `automil init <path>` | Create a new project |
| `automil submit --node <id> --desc "..." --files <f>` | Queue an experiment |
| `automil rank` | Show top-ranked proposals |
| `automil propose --parent <id> --desc "..."` | Add a proposal |
| `automil reconcile` | Sync graph with orchestrator |
| `automil status` | Show experiment summary |
| `automil start-loop` | Enable continuous loop |
| `automil stop-loop` | Allow agent to stop |
| `automil orchestrator start/stop/status` | Manage GPU scheduler |
| `automil viz start/stop/status` | Manage 3D dashboard |

## Examples

See `examples/` for complete worked examples:
- `ovarian_hrd/` - Binary classification with 189 experiments
- `clwd/` - Multi-class lung adenocarcinoma subtyping
- `placeholder/` - Minimal template to start from
