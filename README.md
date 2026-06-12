<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-90%20passed-brightgreen.svg" alt="90 tests passed">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/code%20lines-10,000-ff69b4.svg" alt="10k lines">
</p>

<h1 align="center">Miniclaude</h1>

<p align="center"><strong>A Production-Grade AI Agent Runtime Built from First Principles</strong></p>

<p align="center">
  <em>CLI + Daemon architecture · JSON-RPC 2.0 over TCP NDJSON · ReAct Agent Loop<br>
  Real-time EventBus · Textual TUI · Autonomous Planning · Three-layer Memory<br>
  Tool Security · Context Governance · Skills · Subagents · MCP Integration</em>
</p>

---

## Why Miniclaude?

Most AI agent demos are single-process scripts that call an API directly. **Miniclaude is different.** It's built from Day One as a **production-grade agent runtime** with:

- **Process isolation** — TUI crashes don't kill agent tasks. Multiple frontends (CLI, TUI, Web) share one daemon.
- **Full observability** — Every request, response, tool call, and LLM interaction is traced and replayable.
- **Real-time streaming** — EventBus pushes events to all connected clients simultaneously.
- **Memory across sessions** — Session → Thread → Notes hierarchy persists context across runs.
- **Security by design** — Three-layer tool safety: parameter validation, risk-graded permission approval, failure classification with retry.

## Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│  CLI     │   │  TUI     │   │  Web     │   ← Multiple frontends
│ (mini)   │   │(mini-tui)│   │(future)  │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │  TCP NDJSON + JSON-RPC 2.0
              ┌─────┴─────┐
              │  Daemon   │  ← Single backend process
              │(mini-core)│
              └─────┬─────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
┌────┴────┐  ┌──────┴──────┐  ┌───┴───┐
│EventBus │  │ AgentRunner │  │ Memory│  ← Core subsystems
│+ IPC    │  │ + ReActLoop │  │ Store  │
└─────────┘  └─────────────┘  └───────┘
```

## Quick Start

```bash
# 1. Install
cd miniclaude
pip install -e ".[dev,tui]"

# 2. Set API key
# PowerShell:
$env:LLM_API_KEY="your-api-key"

# 3. Launch daemon (Terminal 1)
mini-core

# 4. Run agent task (Terminal 2)
mini run "Create a Flask app with a /api/users endpoint"

# 5. Launch TUI (Terminal 3)
mini-tui
```

## Feature Overview

### S0 — Protocol Foundation
TCP NDJSON + JSON-RPC 2.0. CLI and daemon are separate processes. Multi-client support with graceful shutdown.

### S1 — Agent Closed Loop
ReAct pattern: `Reason → Act → Observe → Repeat`. LLM calls, tool execution, event logging to `events/{run_id}.jsonl`.

### S2 — Real-time EventBus
Publish/subscribe event system. Events stream to all connected clients simultaneously. IPC push protocol for multi-client event subscription.

### S3 — Autonomous Planning + Trace
LLM-powered task decomposition into DAG of subtasks. Plan executor with dependency ordering. Three-layer trace system (IPC → EventBus → LLM) with SQLite storage and replay.

### S4 — Session Memory
Session → Thread → Notes hierarchy. Auto-extraction of user preferences, project context, learnings, and decisions from completed conversations. Cross-session memory recall injected into system prompts.

### S5 — Tool Security
Three-layer defense: Parameter validation (Schema + Semantics + Path safety) → Risk-graded permission approval (safe/low/medium/high/critical) → Failure classification with exponential backoff retry.

### S6 — Context Governance
Token counting with watermark detection. Smart tool result truncation that preserves errors and structure. LLM-powered conversation compaction with auto-created Notes.

### S7 — Extensions Platform
Skills system with SKILL.md loader and keyword/explicit matching. SubAgent spawner for parallel task execution with isolated contexts. MCP (Model Context Protocol) client with stdio transport and ToolRegistry bridge.

## Project Structure

```
miniclaude/
├── mini_core/          # Core engine (53 modules)
│   ├── agent/          # AgentLoop, Runner, Planner, TaskDAG
│   ├── context/        # TokenCounter, Watermark, Truncator, Compactor
│   ├── events/         # EventBus (15 event types), IPC Subscriber
│   ├── llm/            # OpenAI-compatible LLM Provider
│   ├── mcp/            # MCP Client (stdio), Bridge, ServerConfig
│   ├── memory/         # Session, Thread, Notes, Recall, SQLite Store
│   ├── security/       # Validator, Permissions, Rules, Risk, Failure
│   ├── skills/         # SkillLoader, Registry, Matcher
│   ├── subagents/      # Manager, Spawner, Context Builder
│   ├── tools/          # Base, Registry, 4 built-in tools
│   └── trace/          # Collector, Storage, Replayer
├── mini_cli/           # CLI client
├── mini_tui/           # Textual TUI (7 widgets)
├── skills/             # Example skills (code-review, write-tests, git-helper)
├── tests/              # 90 tests across 8 test files
└── pyproject.toml      # Zero mandatory dependencies (stdlib only)
```

## Test Suite

```bash
$ pytest tests/ -v

tests/test_s0_pingpong.py         7 passed   # Protocol & transport
tests/test_s1_agent_loop.py        9 passed   # ReAct loop & tool calls
tests/test_s2_eventbus.py         10 passed   # EventBus & IPC
tests/test_s3_planning_trace.py   10 passed   # Planning & trace
tests/test_s4_memory.py           10 passed   # Session/thread/notes
tests/test_s5_security.py         14 passed   # Validation, permissions, retry
tests/test_s6_context.py          14 passed   # Watermark, truncation, compact
tests/test_s7_extensions.py       16 passed   # Skills, subagents, MCP
────────────────────────────────────────
Total:                            90 passed
```

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `LLM_API_KEY` | (required) | Your LLM API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | API endpoint (auto-detects DeepSeek/Anthropic) |
| `LLM_MODEL` | `deepseek-v4-flash` | Model name |

## CLI Commands

| Command | Description |
|---------|-------------|
| `mini` | Ping daemon, show status |
| `mini run "goal"` | Execute agent task |
| `mini run --stream "goal"` | Execute with real-time event stream |
| `mini events <run_id>` | Show event log for a run |
| `mini trace list` | List all trace records |
| `mini trace get <run_id>` | Full trace report (JSON) |
| `mini session create --name X` | Create persistent session |
| `mini notes list --session X` | List session memory notes |
| `mini skills list` | List available skills |

## Key Design Decisions

- **No HTTP** — Raw TCP NDJSON for minimal overhead and true streaming
- **stdlib-first** — Only 2 runtime dependencies: `aiohttp` and `textual`
- **Real-time persistence** — Events flushed to disk immediately, crash-safe
- **Decoupled EventBus** — Bus doesn't know about IPC; a separate subscriber manager bridges events to TCP clients
- **Extensible tools** — MCP protocol support lets any MCP-compatible server become a tool

## License

MIT
