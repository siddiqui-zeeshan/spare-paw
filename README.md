# spare-paw

A 24/7 personal AI agent running on a rooted Android phone via Termux, accessible through Telegram. Features multi-model routing via OpenRouter, DAG-based lossless context management, shell and filesystem tools, scheduled tasks, voice transcription, and full-text search over conversation history. Cold starts in ~1 second.

## Features

- **Multi-model routing** -- configure model slots (default, smart, cron) via OpenRouter; switch on the fly
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
- **Agent orchestration** -- spawn multiple subagents in a single turn; agents spawned in the same tool-call batch are deterministically grouped (via a shared batch group_id injected by the tool loop) and their results are delivered together as one synthesized response. Three predefined archetypes: `researcher` (web search + scraping), `coder` (shell + files), `analyst` (data analysis), each with preset tools and system prompt. Safety limits: max 3 concurrent agents, max 3 per group
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
| `context` | `max_messages`, `token_budget`, `safety_margin`, `fresh_tail_count`, `leaf_chunk_size`, `condensed_min_fanout`, `summary_model` |
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

LCM (context compaction) settings:

```yaml
context:
  fresh_tail_count: 32
  leaf_chunk_size: 8
  condensed_min_fanout: 4
  summary_model: "google/gemini-3.1-flash-lite"
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

1. Install `gh` CLI on Termux: `pkg install gh`
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
  +-- TelegramBackend (bot/backend.py)    <-- current implementation
  |     python-telegram-bot, owner auth, photo/voice handling
  |
  v
Core Engine (core/engine.py)
  |  message processing, tool loop, agent orchestration
  |
  +-- core/prompt.py    system prompt assembly
  +-- core/voice.py     Groq Whisper transcription
  +-- core/commands.py  slash command logic
  |
  v
Context Manager (SQLite + FTS5 + DAG-based lossless context management)
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
  |
Cron Scheduler (APScheduler, SQLite-persisted, semaphore-gated)
```

The core engine is decoupled from Telegram via the `MessageBackend` protocol (`backend.py`). `TelegramBackend` in `bot/backend.py` is the current implementation; additional backends (webhook, Discord, etc.) can be added by implementing `MessageBackend` and `IncomingMessage`. `gateway.py` interacts with the backend through `AppState.backend` rather than directly with the Telegram `Application`.

Key design points:

- Single async event loop with a ProcessPoolExecutor (4 workers) for blocking operations
- `asyncio.Semaphore(1)` serializes all model API calls to prevent races
- Heartbeat file touched every 30s; watchdog restarts if stale beyond 90s
- Cron outputs are delivered to Telegram but do not enter conversation memory
- Subagents don't message the user directly; results flow back through a group callback queue, letting the main LLM synthesize a unified response. Results are stored in conversation memory for follow-up questions
- Multiple agents spawned in one tool-call batch are deterministically grouped by the tool loop (shared batch group_id); the turn stop is deferred until all spawns in the batch complete
- Safety limits: max 3 concurrent agents, max 3 per group
- Three agent archetypes (`researcher`, `coder`, `analyst`) each set appropriate tools and system prompt
- Each agent tracks token usage (prompt, completion, total) from OpenRouter API responses
- Token counting uses tiktoken with a configurable safety margin for non-OpenAI models
- DAG compaction runs automatically after each turn: messages beyond the fresh tail are chunked into leaf summaries, and when enough leaves accumulate they condense into higher-level nodes. Schema v3 adds a `summary_nodes` table with FTS5 index. `assemble()` injects compressed history between the system prompt and fresh messages

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
  backend.py         # MessageBackend protocol + IncomingMessage dataclass
  config.py          # Config loading
  db.py              # SQLite connection, schema, migrations
  context.py         # Sliding window context + FTS5 search
  gateway.py         # Main async loop (uses AppState.backend)
  setup_wizard.py    # Interactive onboarding
  core/
    engine.py        # Message processing and tool loop (backend-agnostic)
    prompt.py        # System prompt builder
    voice.py         # Groq Whisper transcription
    commands.py      # Slash command logic
  bot/
    backend.py       # TelegramBackend: implements MessageBackend for Telegram
    handler.py       # Telegram update handler with queue
    commands.py      # Telegram command wiring
    voice.py         # Voice message handling
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
