# spare-paw

A 24/7 personal AI agent accessible through Telegram. Runs on macOS, Linux, Windows, Android (Termux), or Docker. Features role-based model selection via OpenRouter (7 roles with fallback chain), DAG-based lossless context management, shell and filesystem tools, scheduled tasks, voice transcription, and full-text search over conversation history. Cold starts in ~1 second.

## Features

- **Role-based model selection** -- 7 model roles (main_agent, coder, planner, cron, researcher, analyst, summary) each independently configurable via OpenRouter; fallback chain: role-specific model -> main_agent -> google/gemini-2.0-flash
- **Tool use** -- shell commands, file operations, web search, web scraping, cron management; all exposed as LLM function calls
- **Scheduled tasks (cron)** -- create, edit, pause, resume, and manage recurring AI tasks with per-cron model selection
- **One-shot reminders** -- ask the bot to remind you of something in X minutes/hours; it creates a cron that fires once and auto-deletes itself (e.g. "remind me to call John in 30 minutes")
- **Photo/image support** -- send photos via Telegram; they are base64-encoded and forwarded to the model as multimodal vision messages. Caption is used as the prompt (defaults to "What do you see in this image?")
- **Voice messages** -- Groq Whisper transcription for Telegram voice notes
- **Prompt files** -- loads `IDENTITY.md`, `USER.md`, and `SYSTEM.md` from `~/.spare-paw/` on every turn for personality, user preferences, and device context. Editable live without restart
- **Full-text search** -- FTS5-backed search across all conversation history
- **DAG-based lossless context management (LCM)** -- when conversation history grows beyond the fresh tail (32 messages), older messages are automatically summarized into leaf nodes (~8 messages each); when 4+ leaves accumulate they condense into higher-level summaries. Every original message is preserved and searchable. Summaries are assembled between the system prompt and fresh messages so the LLM retains awareness of older context. Compaction uses a cheap configurable model (default `google/gemini-3.1-flash-lite`) to keep costs low
- **LCM tools** -- `lcm_grep` (search raw messages and compressed summaries via FTS5), `lcm_expand` (drill into a summary to recover original messages, token-capped), `lcm_describe` (get summaries for a time range)
- **Sliding window context** -- token-budgeted context assembly with configurable window size and safety margin
- **Message queue with backpressure** -- incoming messages queue while the bot is busy; typing indicator signals processing
- **Heartbeat watchdog** -- detects event loop starvation and deadlocks, not just process crashes
- **Deep thinking (`/plan`)** -- on-demand planning phase that decomposes complex requests into a structured execution plan before the tool loop runs. A single cheap LLM call (no tools) produces a step-by-step plan with tool/agent classification and parallelism hints; the plan is injected as context for the main model to follow. Regular messages skip planning entirely (zero overhead)
- **Agent orchestration** -- spawn multiple subagents in a single turn; agents spawned in the same tool-call batch are deterministically grouped (via a shared batch group_id injected by the tool loop) and their results are delivered together as one synthesized response. Three predefined archetypes: `researcher` (web search + scraping), `coder` (shell + files), `analyst` (data analysis), each with preset tools and system prompt. Safety limits: max 3 concurrent agents, max 3 per group
- **Token/cost tracking** -- per-agent token usage tracking (prompt, completion, total) from OpenRouter, visible via `list_agents`
- **MCP client** -- connect to external MCP servers (GitHub, filesystem, etc.) and use their tools alongside native tools
- **Owner-only auth** -- all messages from non-owner Telegram users are silently ignored
- **Platform-aware defaults** -- auto-detects macOS, Linux, Windows, and Termux at startup; sets appropriate shell descriptions, allowed paths, and prompt files

## Prerequisites

- Python 3.11+
- API keys: OpenRouter (required), Telegram Bot Token (required), Tavily (optional, for web search), Groq (optional, for voice)
- Node.js (optional, for MCP servers that use `npx`)

## Quick Start

### One-liner install

```bash
curl -sSL https://raw.githubusercontent.com/siddiqui-zeeshan/spare-paw/main/scripts/install.sh | bash
```

This installs spare-paw in a venv at `~/.spare-paw/venv/`, runs the setup wizard, and optionally creates a systemd service (Linux). Works on Linux, macOS, and Termux.

### Manual install

```bash
git clone https://github.com/siddiqui-zeeshan/spare-paw.git
cd spare-paw
pip install .
```

After installation, the `spare-paw` CLI entry point is available:

```bash
spare-paw setup    # interactive setup wizard
spare-paw gateway  # start the bot
```

You can also invoke via `python -m spare_paw setup` / `python -m spare_paw gateway` without installing.

### Termux (Android)

```bash
pkg update && pkg upgrade
pkg install python python-pip git

git clone <repo-url>
cd spare-paw
pip install --break-system-packages -e .
```

Use `termux-wake-lock` before starting to prevent Android from killing the process.

### Docker

```bash
docker build -t spare-paw .
docker run -d \
  -v ~/.spare-paw:/root/.spare-paw \
  -p 8080:8080 \
  spare-paw
```

The container mounts `~/.spare-paw` for persistent config, database, and logs. Port 8080 is used when running the webhook backend.

### Setup

Run the interactive setup wizard to create `~/.spare-paw/config.yaml`, initialize the database, and copy platform-appropriate prompt files:

```bash
spare-paw setup
```

The wizard detects your platform (Termux, macOS, Linux, or Windows) and writes platform-appropriate defaults — allowed file paths, shell tool description, and `SYSTEM.md` — into `~/.spare-paw/`.

### Running

```bash
# Start the gateway
spare-paw gateway

# With watchdog (recommended for always-on operation on Termux)
bash scripts/watchdog.sh
```

## Configuration

Config file location: `~/.spare-paw/config.yaml`

A template is provided at `config.example.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `backend` | `"telegram"` (default) or `"webhook"` |
| `telegram` | Bot token and owner ID |
| `webhook` | Port, optional secret, and `enabled` flag for the HTTP webhook backend |
| `remote` | `url` and `secret` for `spare-paw chat` to reach a remote instance |
| `openrouter` | OpenRouter API key |
| `models` | Role-based model assignments: `main_agent`, `coder`, `planner`, `cron`, `researcher`, `analyst`, `summary` |
| `groq` | Groq API key for voice transcription |
| `tavily` | Tavily API key for web search |
| `context` | `max_messages`, `token_budget`, `safety_margin`, `fresh_tail_count`, `leaf_chunk_size`, `condensed_min_fanout` |
| `tools` | Per-tool enable/disable, timeouts, allowed paths |
| `mcp` | MCP client server connections |
| `agent` | `max_tool_iterations`, `system_prompt` template |
| `logging` | Log level, rotation size, backup count |

Role-based model configuration:

```yaml
models:
  main_agent: "google/gemini-2.0-flash"       # primary model for conversations
  coder: "google/gemini-2.5-pro"               # used by coder subagents
  planner: "anthropic/claude-sonnet-4"         # used by /plan deep thinking
  cron: "google/gemini-2.0-flash"              # used for cron job execution
  researcher: "google/gemini-2.0-flash"        # used by researcher subagents
  analyst: "google/gemini-2.0-flash"           # used by analyst subagents
  summary: "google/gemini-3.1-flash-lite"      # used for LCM context summaries
```

Each role falls back through the chain: role-specific model -> `main_agent` -> `google/gemini-2.0-flash`. You only need to set the roles you want to override.

LCM (context compaction) settings:

```yaml
context:
  fresh_tail_count: 32
  leaf_chunk_size: 8
  condensed_min_fanout: 4
```

### Webhook backend

Set `backend: "webhook"` to replace Telegram with a plain HTTP server. Useful for Docker deployments, CI testing, or custom integrations:

```yaml
backend: "webhook"
webhook:
  port: 8080
  secret: "your-secret-here"   # optional; sent as Bearer token
```

**Send a message:**

```bash
curl -X POST http://localhost:8080/message \
  -H "Authorization: Bearer your-secret-here" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

**Poll for responses:**

```bash
curl "http://localhost:8080/poll?timeout=30" \
  -H "Authorization: Bearer your-secret-here"
```

The poll endpoint returns a JSON array of response messages and supports images and voice (base64-encoded in the request body).

## TUI (`spare-paw chat`)

`spare-paw chat` gives you a full-screen terminal interface to the same engine, without opening Telegram.

### Install optional dependencies

```bash
pip install spare-paw[tui]   # rich + textual
```

### Modes

| Invocation | Behaviour |
|---|---|
| `spare-paw chat` | Full-screen Textual TUI (message log, input bar, keyboard shortcuts) |
| `echo "question" \| spare-paw chat` | Pipe / non-interactive: reads stdin, prints reply, exits |

### Remote client mode

When `remote.url` is set in `~/.spare-paw/config.yaml`, `spare-paw chat` connects to a running spare-paw instance (e.g. the bot on your phone) via its webhook API instead of starting a local engine:

```yaml
remote:
  url: "http://192.168.1.10:8080"
  secret: "your-webhook-secret"   # matches webhook.secret on the server
```

The client health-checks the server on startup and exits with an error message if the server is unreachable.

### Standalone local mode

When `remote.url` is absent, `spare-paw chat` spins up a full local engine (same tools, context, models as the gateway). Useful for testing or when the phone isn't available.

### TUI keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+C` | Exit |
| `Ctrl+L` | Clear log |
| `Ctrl+N` | New conversation (`/forget`) |
| `F1` | Help |

### TUI visual features

The TUI renders a polished full-screen interface:

- **Streaming output** -- tokens appear as they arrive from the model, word by word
- **Message timestamps** -- each user and bot message shows the time in HH:MM AM/PM format
- **Full-width turn dividers** -- Rich Rule horizontal rules spanning the full terminal width separate conversation turns for readability
- **Tool call panels** -- tool calls render in bordered panels instead of inline dim text
- **Cat-themed thinking verbs** -- while waiting for the first token, a rotating cat-themed verb is shown (e.g. "Purring...", "Pawing...", "Whisker-twitching...", "Tail-flicking...") instead of a plain "Thinking..." indicator
- **Conversation history on startup** -- the last 10 messages are loaded from the server when the TUI starts, so the chat continues seamlessly from where you left off
- **Slash command autocomplete** -- typing `/` in the input bar shows a filtered list of available commands; press Tab or Enter to complete
- **Status bar** -- bottom bar shows connection status with a colored dot, active model name, message count, and tool call count
- **Styled input** -- input field has a prompt prefix for visual clarity

### Tool call visibility

Tool calls are rendered as translucent bordered panels that appear while the model is working and disappear once the response arrives.

### Dual-backend: Telegram + webhook simultaneously

The gateway can run Telegram and the webhook API at the same time. This lets `spare-paw chat` coexist with the Telegram bot on the same instance:

```yaml
backend: "telegram"
webhook:
  enabled: true
  port: 8080
  secret: "your-secret-here"
```

The webhook API exposes `POST /message`, `GET /poll`, `GET /stream` (SSE), `GET /health`, and `GET /history`. The `/stream` endpoint delivers messages and tool events in real-time via Server-Sent Events.

**Fetch conversation history:**

```bash
curl "http://localhost:8080/history?limit=20" \
  -H "Authorization: Bearer your-secret-here"
```

Returns a JSON array of recent messages (newest last). `limit` defaults to 50.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/cron list` | List all crons with ID, schedule, next run, model, status |
| `/cron remove <id>` | Delete a cron |
| `/cron pause <id>` | Pause a cron without deleting |
| `/cron resume <id>` | Resume a paused cron |
| `/cron info <id>` | Show details, last result, recent failures |
| `/config show` | Show current runtime config |
| `/config reset` | Reset overrides to config.yaml defaults |
| `/status` | Uptime, memory, DB size, active crons, last error |
| `/search <query>` | Full-text search over conversation history |
| `/forget` | Start a new conversation (history preserved in DB) |
| `/model` | Show all role-to-model assignments |
| `/model <model_id>` | Set the main_agent model |
| `/model <role> <model_id>` | Set a specific role's model (e.g. `/model coder google/gemini-2.5-pro`) |
| `/models` | List available models from OpenRouter |
| `/models <filter>` | Filter available models by keyword (e.g. `/models gemini`) |
| `/plan <prompt>` | Deep thinking: plan before executing (decomposes into steps, then runs) |
| `/mcp` | List connected MCP servers and their tools |

## Tools

All tools are exposed to the LLM as callable functions. Blocking tools run in a `ProcessPoolExecutor` to keep the async event loop responsive.

| Tool | Description | Notes |
|------|-------------|-------|
| `shell` | Execute shell commands | Platform-aware description; configurable timeout (default 30s), output truncated at 10K chars |
| `files` | Read, write, append, list, delete files | Restricted to `allowed_paths` in config; defaults set per platform |
| `tavily_search` | Web search via Tavily API | Optional; requires Tavily API key |
| `web_scrape` | Fetch and extract content from a URL | BeautifulSoup parsing, 15s timeout, 20K char limit |
| `cron_create` | Create a scheduled task | Accepts name, cron expression, prompt, optional model |
| `cron_edit` | Edit an existing scheduled task | Update name, schedule, prompt, and/or model by cron ID |
| `cron_delete` | Delete a scheduled task | By cron ID |
| `spawn_agent` | Spawn a subagent for parallel work | `agent_type`: `researcher`, `coder`, or `analyst`; multiple can be spawned in one turn and are auto-grouped |
| `list_agents` | List running/completed agents | Shows status, result preview, and per-agent token usage (prompt, completion, total) |
| `cron_list` | List all scheduled tasks | Returns schedule, status, last run info |
| `lcm_grep` | Search raw messages and compressed summaries | FTS5 full-text search across history and DAG summaries |
| `lcm_expand` | Drill into a summary node | Recovers original messages under a summary, token-capped |
| `lcm_describe` | Get summaries for a time range | Returns DAG summary nodes covering the specified period |

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

## GitHub Integration

The bot can interact with GitHub autonomously using the `gh` CLI through its existing shell tool -- no MCP server or Node.js dependency required.

### Setup

1. Install `gh` CLI (`pkg install gh` on Termux, `brew install gh` on macOS, or via the [official installer](https://cli.github.com/) on other platforms)
2. Create a fine-grained PAT on GitHub with the scopes you need (repo contents, issues, PRs)
3. Authenticate: `echo "<token>" | gh auth login --with-token`
4. Wire up git credentials: `gh auth setup-git`

That's it. The bot picks up `gh` like any other shell command.

### What the bot can do

- **Browse repos** -- list issues, PRs, and repo metadata via `gh issue list`, `gh pr list`, etc.
- **End-to-end code changes** -- clone a repo, create a feature branch, make edits, commit, push, and open a PR, all from a single Telegram message
- **Review PRs** -- fetch the branch, read the diff, cross-check against the codebase, and leave review comments via `gh pr review`
- **Background agents for heavy tasks** -- spawns a `coder` subagent for complex work like PR reviews so the main Telegram loop stays responsive
- **Persistent memory** -- the git auth workflow is remembered across sessions via the DAG-based context system, so setup is a one-time step

### Example prompts

```
List open issues on siddiqui-zeeshan/spare-paw
Fix issue #11 in our repo — it's about adding a dependency lockfile
Review PR #13 and leave comments
```

### Real example: autonomous issue fix + PR review

When asked to fix a known issue ("no dependency lockfile"), the bot autonomously:

1. Installed `gh` CLI on its own (`pkg install gh`)
2. Listed issues via `gh issue list`
3. Cloned the repo and created branch `fix/add-dependency-lockfile`
4. Read `pyproject.toml`, ran `pip-compile` to generate `requirements.txt`
5. Committed, pushed, and created [PR #13](https://github.com/siddiqui-zeeshan/spare-paw/pull/13)
6. Then when asked to review its own PR, spawned a background agent that:
   - Fetched the PR branch and read the lockfile
   - Cross-checked Python imports across the codebase against the lockfile
   - Left a review comment: *"Looks good. One minor observation: ensure pytest and pytest-asyncio are included if you intend to run tests in the CI pipeline."*

All from two Telegram messages. No human intervention after the initial prompt.

### More real-world examples

**1. Parallel multi-agent research**

When asked to research AI model pricing and the Anthropic changelog simultaneously, the bot:

1. Spawned 2 background agents in a single turn (`researcher` archetype for both)
2. Both ran in parallel -- one searched OpenRouter pricing, the other scraped `docs.anthropic.com`
3. Both spawn_agent calls were in the same tool-call batch, so the tool loop assigned them the same group_id
4. Results were bundled and synthesized by the main LLM into one coherent response
5. User received a single Telegram message covering both topics, not two separate dumps

**2. Self-creating tools + cron automation**

When asked to monitor system health, the bot:

1. Used `tool_create` to write a new `check_system_health` shell script on the fly (checking battery, RAM, storage)
2. Sent an approval button via Telegram inline keyboard -- user tapped "Approve"
3. Created a cron job (`0 */3 * * *`) to run the health check every 3 hours
4. The tool and cron persist across restarts -- zero human code involved

**3. LCM memory recall**

After 100+ messages in a conversation, the bot had compressed history into 12 DAG summaries (11 leaves + 1 condensed). When later asked "what did we talk about earlier?":

1. Used `lcm_grep` to search compressed summaries via FTS5
2. Used `lcm_expand` to drill into matching summary nodes and recover original messages
3. Recalled specific details from early in the conversation that would have been lost with a flat sliding window

## Architecture

```
MessageBackend (protocol)
  |
  +-- TelegramBackend (bot/backend.py)     -- Telegram bot, owner auth, photo/voice handling
  |
  +-- WebhookBackend (webhook/backend.py)  -- HTTP server (POST /message, GET /poll, GET /health)
  |
  v
Core Engine (core/engine.py)
  |  message processing, tool loop, agent orchestration
  |
  +-- core/prompt.py    system prompt assembly
  +-- core/planner.py   deep thinking planning phase (/plan)
  +-- core/voice.py     Groq Whisper transcription
  +-- core/commands.py  slash command logic
  |
  v
Context Manager (SQLite + FTS5 + DAG-based lossless context management)
  |
  v
Model Router (OpenRouter API, role-based selection, semaphore-serialized, retry with backoff)
  |
  v
Tools (ProcessPoolExecutor: shell, files, web search, web scrape, cron, vision)
  |                         \
  |                          MCP Client (connects to external MCP servers)
  |
Agent Orchestrator (spawn_agent -> parallel multi-agent spawning)
  |
Cron Scheduler (APScheduler, SQLite-persisted, semaphore-gated)
```

The core engine is decoupled from any specific frontend via the `MessageBackend` protocol (`backend.py`). `TelegramBackend` in `bot/backend.py` and `WebhookBackend` in `webhook/backend.py` are the two current implementations. Additional backends (Discord, Slack, etc.) can be added by implementing `MessageBackend` and `IncomingMessage`. `gateway.py` interacts with the backend through `AppState.backend` rather than directly with any platform SDK.

Key design points:

- Single async event loop with a ProcessPoolExecutor (4 workers) for blocking operations
- `asyncio.Semaphore(1)` serializes all model API calls to prevent races
- Heartbeat file touched every 30s; watchdog restarts if stale beyond 90s
- Cron outputs are delivered to the active backend but do not enter conversation memory
- Subagents don't message the user directly; results flow back through a group callback queue, letting the main LLM synthesize a unified response. Results are stored in conversation memory for follow-up questions
- Multiple agents spawned in one tool-call batch are deterministically grouped by the tool loop (shared batch group_id); the turn stop is deferred until all spawns in the batch complete
- Safety limits: max 3 concurrent agents, max 3 per group
- Three agent archetypes (`researcher`, `coder`, `analyst`) each set appropriate tools and system prompt
- Each agent tracks token usage (prompt, completion, total) from OpenRouter API responses
- Token counting uses tiktoken with a configurable safety margin for non-OpenAI models
- DAG compaction runs automatically after each turn: messages beyond the fresh tail are chunked into leaf summaries, and when enough leaves accumulate they condense into higher-level nodes. Schema v3 adds a `summary_nodes` table with FTS5 index. `assemble()` injects compressed history between the system prompt and fresh messages
- Platform detection at startup (`platform.py`) sets shell tool description, default allowed paths, and selects the correct `SYSTEM.md` template for Termux, macOS, Linux, and Windows

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
  __main__.py        # Entry point: spare-paw setup / spare-paw gateway / spare-paw chat
  backend.py         # MessageBackend protocol + IncomingMessage dataclass
  config.py          # Config loading
  db.py              # SQLite connection, schema, migrations
  context.py         # Sliding window context + FTS5 search
  gateway.py         # Main async loop (uses AppState.backend)
  platform.py        # Platform detection and platform-appropriate defaults
  setup_wizard.py    # Interactive onboarding (platform-aware)
  core/
    engine.py        # Message processing and tool loop (backend-agnostic)
    prompt.py        # System prompt builder
    voice.py         # Groq Whisper transcription
    commands.py      # Slash command logic
    planner.py       # Deep thinking planning phase (/plan)
  bot/
    backend.py       # TelegramBackend: implements MessageBackend for Telegram
    handler.py       # Telegram update handler with queue
    commands.py      # Telegram command wiring
    voice.py         # Voice message handling
  webhook/
    backend.py       # WebhookBackend: HTTP server (POST /message, GET /poll, GET /stream, GET /health)
  cli/
    entry.py         # spare-paw chat entry point (remote / local / pipe dispatch)
    client.py        # RemoteClient: HTTP client for the webhook API (send, poll, stream, history)
    pipe.py          # Pipe / non-interactive mode
  tui/
    app.py           # Textual TUI app (SparePawTUI) and TUIBackend
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
    lcm_tools.py     # LCM tools (lcm_grep, lcm_expand, lcm_describe)
  cron/
    scheduler.py     # APScheduler setup
    executor.py      # Cron job execution
  mcp/
    client.py        # MCP client: connects to external MCP servers
    schema.py        # MCP schema conversion utilities
defaults/
  IDENTITY.md        # Default bot personality (generic)
  IDENTITY.termux.md # Bot personality variant for Termux/Android
  SYSTEM.md          # Default SYSTEM.md (Termux)
  SYSTEM.mac.md      # macOS system context
  SYSTEM.linux.md    # Linux system context
  USER.md            # User preferences template
```

## Known Issues

- ~~**MarkdownV2 rendering falls back to plain text**~~ — Fixed: responses are now converted from standard Markdown to Telegram HTML format (`<b>`, `<i>`, `<code>`, `<pre>`, `<a>`) with plain text fallback.
- **Shell commands logged verbatim** — Commands containing secrets (API keys, tokens) appear in log files.
- **owner_id type mismatch risk** — If config has owner_id as a string, auth comparison silently fails. Should cast to int.
- **No dependency lockfile** — Builds use latest compatible versions, not pinned.
- **No prompt injection protection on web_scrape** — Scraped content enters the model context without sanitization. Malicious pages could inject instructions.
- **Key security rules not repeated at end of system prompt** — Model attention is strongest at start/end of prompt; security constraints should be reinforced at both.

## License

MIT — see [LICENSE](LICENSE).
