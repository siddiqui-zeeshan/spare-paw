# spare-paw

A 24/7 personal AI agent running on a rooted Android phone via Termux, accessible through Telegram. Features multi-model routing via OpenRouter, shell and filesystem tools, scheduled tasks, voice transcription, and full-text search over conversation history. Cold starts in ~1 second.

## Features

- **Multi-model routing** -- configure model slots (default, smart, cron) via OpenRouter; switch on the fly
- **Tool use** -- shell commands, file operations, web search, web scraping, cron management; all exposed as LLM function calls
- **Scheduled tasks (cron)** -- create, edit, pause, resume, and manage recurring AI tasks with per-cron model selection
- **Photo/image support** -- send photos via Telegram; they are base64-encoded and forwarded to the model as multimodal vision messages. Caption is used as the prompt (defaults to "What do you see in this image?")
- **Voice messages** -- Groq Whisper transcription for Telegram voice notes
- **Prompt files** -- loads `IDENTITY.md`, `USER.md`, and `SYSTEM.md` from `~/.spare-paw/` on every turn for personality, user preferences, and device context. Editable live without restart
- **Full-text search** -- FTS5-backed search across all conversation history
- **Sliding window context** -- token-budgeted context assembly with configurable window size and safety margin
- **Message queue with backpressure** -- incoming messages queue while the bot is busy; typing indicator signals processing
- **Heartbeat watchdog** -- detects event loop starvation and deadlocks, not just process crashes
- **Agent orchestration** -- spawn multiple subagents in a single turn; agents spawned in the same batch are auto-grouped and their results are delivered together as one synthesized response. Three predefined archetypes: `researcher` (web search + scraping), `coder` (shell + files), `analyst` (data analysis), each with preset tools and system prompt. Safety limits: max 3 concurrent agents, max 3 per group, 30-second rate limit between separate spawn requests
- **Token/cost tracking** -- per-agent token usage tracking (prompt, completion, total) from OpenRouter, visible via `list_agents`
- **MCP client** -- connect to external MCP servers (GitHub, filesystem, etc.) and use their tools alongside native tools
- **Owner-only auth** -- all messages from non-owner Telegram users are silently ignored

## Prerequisites

- Python 3.11+
- Termux (on Android)
- Rooted Android recommended (8GB+ RAM, 128GB+ storage, always-on Wi-Fi)
- API keys: OpenRouter (required), Telegram Bot Token (required), Tavily (optional, for web search), Groq (optional, for voice)
- Node.js (optional, for MCP servers that use `npx`)

## Quick Start

### Termux installation

```bash
pkg update && pkg upgrade
pkg install python python-pip git

git clone <repo-url>
cd spare-paw
pip install --break-system-packages -e .
```

### Setup

Run the interactive setup wizard to create `~/.spare-paw/config.yaml`, validate API keys, and initialize the database:

```bash
python -m spare_paw setup
```

### Running

```bash
# Start the gateway
python -m spare_paw gateway

# With watchdog (recommended for always-on operation)
bash scripts/watchdog.sh
```

Use `termux-wake-lock` before starting to prevent Android from killing the process.

## Configuration

Config file location: `~/.spare-paw/config.yaml`

A template is provided at `config.example.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `telegram` | Bot token and owner ID |
| `openrouter` | OpenRouter API key |
| `models` | Model slots: `default`, `smart`, `cron_default` |
| `groq` | Groq API key for voice transcription |
| `tavily` | Tavily API key for web search |
| `context` | `max_messages`, `token_budget`, `safety_margin` |
| `tools` | Per-tool enable/disable, timeouts, allowed paths |
| `mcp` | MCP client server connections |
| `agent` | `max_tool_iterations`, `system_prompt` template |
| `logging` | Log level, rotation size, backup count |

Model slot examples:

```yaml
models:
  default: "google/gemini-2.0-flash"
  smart: "anthropic/claude-sonnet-4"
  cron_default: "google/gemini-2.0-flash"
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/cron list` | List all crons with ID, schedule, next run, model, status |
| `/cron remove <id>` | Delete a cron |
| `/cron pause <id>` | Pause a cron without deleting |
| `/cron resume <id>` | Resume a paused cron |
| `/cron info <id>` | Show details, last result, recent failures |
| `/config show` | Show current runtime config |
| `/config model <name>` | Override default model for this session |
| `/config reset` | Reset overrides to config.yaml defaults |
| `/status` | Uptime, memory, DB size, active crons, last error |
| `/search <query>` | Full-text search over conversation history |
| `/forget` | Start a new conversation (history preserved in DB) |
| `/model <name>` | Shortcut for `/config model <name>` |
| `/mcp` | List connected MCP servers and their tools |

## Tools

All tools are exposed to the LLM as callable functions. Blocking tools run in a `ProcessPoolExecutor` to keep the async event loop responsive.

| Tool | Description | Notes |
|------|-------------|-------|
| `shell` | Execute shell commands in Termux | Configurable timeout (default 30s), output truncated at 10K chars |
| `files` | Read, write, append, list, delete files | Restricted to `allowed_paths` in config |
| `tavily_search` | Web search via Tavily API | Optional; requires Tavily API key |
| `web_scrape` | Fetch and extract content from a URL | BeautifulSoup parsing, 15s timeout, 20K char limit |
| `cron_create` | Create a scheduled task | Accepts name, cron expression, prompt, optional model |
| `cron_edit` | Edit an existing scheduled task | Update name, schedule, prompt, and/or model by cron ID |
| `cron_delete` | Delete a scheduled task | By cron ID |
| `spawn_agent` | Spawn a subagent for parallel work | `agent_type`: `researcher`, `coder`, or `analyst`; multiple can be spawned in one turn and are auto-grouped |
| `list_agents` | List running/completed agents | Shows status, result preview, and per-agent token usage (prompt, completion, total) |
| `cron_list` | List all scheduled tasks | Returns schedule, status, last run info |

## MCP (Model Context Protocol)

Connect to any MCP server to pull in additional tools. These tools become available to the LLM alongside the native tools. Configure servers in `config.yaml`:

```yaml
mcp:
  servers:
    - name: "github"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
```

Each server is launched as a subprocess using stdio transport. The `/mcp` Telegram command shows connected servers and their available tools.

### Dependency

MCP support requires `mcp>=1.26.0`, included in the package dependencies.

## Architecture

```
Telegram Bot (python-telegram-bot)
  |
  v
Context Manager (SQLite + FTS5 + sliding window)
  |
  v
Model Router (OpenRouter API, semaphore-serialized, retry with backoff)
  |
  v
Tools (ProcessPoolExecutor: shell, files, web search, web scrape, cron, vision)
  |                         \
  |                          MCP Client (connects to external MCP servers)
  |
Agent Orchestrator (spawn_agent -> parallel multi-agent spawning)
  |                    \
  |                     Auto-grouping: agents spawned in the same batch are grouped;
  |                     turn stop is deferred until all spawns complete.
  |                     Group callback: results flow back via a queue callback,
  |                     synthesized into one coherent response by the main LLM.
  |                     Results stored in conversation memory for follow-ups
  |
Cron Scheduler (APScheduler, SQLite-persisted, semaphore-gated)
```

Key design points:

- Single async event loop with a ProcessPoolExecutor (4 workers) for blocking operations
- `asyncio.Semaphore(1)` serializes all model API calls to prevent races
- Heartbeat file touched every 30s; watchdog restarts if stale beyond 90s
- Cron outputs are delivered to Telegram but do not enter conversation memory
- Subagents don't message the user directly; results flow back through a group callback queue, letting the main LLM synthesize a unified response. Results are stored in conversation memory for follow-up questions
- Multiple agents spawned in one tool-call batch are auto-grouped; the turn stop is deferred until all spawns in the batch complete
- Safety limits: max 3 concurrent agents, max 3 per group, 30-second rate limit between separate spawn requests
- Three agent archetypes (`researcher`, `coder`, `analyst`) each set appropriate tools and system prompt
- Each agent tracks token usage (prompt, completion, total) from OpenRouter API responses
- Token counting uses tiktoken with a configurable safety margin for non-OpenAI models

See [SPEC.md](SPEC.md) for the full specification including database schema, concurrency model, and design decisions.

## Development

### Install dev dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Project structure

```
src/spare_paw/
  __main__.py        # Entry point: setup / gateway
  config.py          # Config loading
  db.py              # SQLite connection, schema, migrations
  context.py         # Sliding window context + FTS5 search
  gateway.py         # Main async loop
  setup_wizard.py    # Interactive onboarding
  bot/
    handler.py       # Message handler with queue
    commands.py      # Telegram commands
    voice.py         # Groq Whisper transcription
  router/
    openrouter.py    # OpenRouter API client
    tool_loop.py     # Tool-use execution loop
  tools/
    registry.py      # Tool registration + JSON schemas
    shell.py         # Shell execution
    files.py         # File operations
    tavily_search.py # Tavily Search API
    web_scrape.py    # URL fetch + parsing
    cron_tools.py    # Cron CRUD tools (create, edit, delete, list)
  cron/
    scheduler.py     # APScheduler setup
    executor.py      # Cron job execution
  mcp/
    client.py        # MCP client: connects to external MCP servers
    schema.py        # MCP schema conversion utilities
```

## DataGrip Database MCP Tools

The `datagrip-database-mcp/` subdirectory contains a JetBrains plugin that extends DataGrip's built-in MCP server with six database tools, letting Claude Code query databases through DataGrip's existing connections (SSH tunnels, SSL, auth) with zero config.

| Tool | Description |
|------|-------------|
| `list_datasources` | List all configured data sources |
| `get_schema` | Get schema metadata for a data source |
| `run_query` | Execute a SQL query and return results |
| `explain_query` | Run EXPLAIN on a query and return the plan |
| `get_table_info` | Get detailed info for a specific table |
| `search_schema` | Search across schema objects by name |

### Build

```bash
cd datagrip-database-mcp
./gradlew build
```

Install the resulting plugin ZIP from `build/distributions/` into DataGrip.

## Known Issues

- **MarkdownV2 rendering falls back to plain text** — Model outputs standard Markdown but Telegram's MarkdownV2 is strict about escaping. Currently falls back to plain text. Fix: convert to Telegram HTML format instead.
- **Shell commands logged verbatim** — Commands containing secrets (API keys, tokens) appear in log files.
- **owner_id type mismatch risk** — If config has owner_id as a string, auth comparison silently fails. Should cast to int.
- **No dependency lockfile** — Builds use latest compatible versions, not pinned.
- **No prompt injection protection on web_scrape** — Scraped content enters the model context without sanitization. Malicious pages could inject instructions.
- **Key security rules not repeated at end of system prompt** — Model attention is strongest at start/end of prompt; security constraints should be reinforced at both.

## License

MIT — see [LICENSE](LICENSE).
