# Agent Compatibility Guide

autoMIL works with any coding agent that can read files, edit code, and run shell commands.

## Claude Code (First-Class Support)

Claude Code is the recommended agent with native skill and hook integration.

### Setup
1. The `.claude/skills/automil.md` skill is included in the repo
2. The `.claude/hooks/on_stop.sh` hook prevents stopping mid-loop
3. Start with: `/automil` in a Claude Code session

### Features
- Automatic loop continuation via stop hook
- Skill-based activation
- Native file editing and shell access

## Cursor

### Setup
1. Open your autoMIL project in Cursor
2. Add `program.md` content to your Cursor rules or system prompt
3. Tell the agent: "Follow the instructions in program.md to run the experiment loop"

### Limitations
- No automatic stop prevention (agent may pause between experiments)
- Manual context management

## Codex / OpenAI Agents

### Setup
1. Include `program.md` in the agent's context
2. Ensure the agent has shell access to run `automil` commands
3. Start with: "Read program.md and begin the experiment loop"

## Aider

### Setup
1. Start aider in the project directory
2. Use `/read program.md` to load instructions
3. Tell it to follow the experiment loop

### Limitations
- Aider is optimized for code editing, not long-running loops
- May need periodic re-prompting

## Windsurf

### Setup
1. Open project in Windsurf
2. Add program.md to Cascade context
3. Start the loop

## Universal Requirements

Any compatible agent must be able to:
1. **Read files** - config.yaml, graph.json, learnings.md, program.md
2. **Edit files** - train.py, model code
3. **Run shell commands** - `automil submit`, `automil rank`, etc.
4. **Maintain context** - Remember what experiments have been tried

## How It Works

The agent interacts with autoMIL entirely through:
- **Files**: config.yaml, graph.json, learnings.md (read), train.py (edit)
- **CLI**: `automil` commands for all operations
- **program.md**: Complete instructions for the experiment loop

No agent-specific APIs or integrations are required beyond basic file and shell access.
