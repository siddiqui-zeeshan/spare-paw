# spare-paw — Complete Specification

## Overview

A 24/7 personal AI agent running on a rooted Android phone via Termux, accessible through Telegram. Features configurable multi-model routing via OpenRouter, automation tools, scheduled tasks, voice transcription, and full-text search over conversation history.

**Target hardware:** Rooted Android, 8GB RAM, 128GB storage, always-on Wi-Fi, no personal data.
**Language:** Python 3.11+
**Interface:** Telegram (owner-only)
**LLM backend:** OpenRouter API (any model)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Telegram Bot                           │
│  (python-telegram-bot · owner-only auth · commands)       │
│  Voice messages → Groq Whisper transcription              │
│  Message queue with backpressure + "thinking..." indicator│
└──────────────┬──────────────────────┬──────────────────┘
               │ user messages        │ cron results
               ▼                      ▲
┌──────────────────────────┐  ┌───────────────────────────┐
│   Context Manager         │  │    Cron Scheduler          │
│  (SQLite · sliding window │  │  (APScheduler · SQLite     │
│   · FTS5 full-text search │  │   persisted · per-cron     │
│   · token budget)         │  │   model · semaphore-gated) │
└──────────┬───────────────┘  └───────────┬───────────────┘
           │ assembled context             │ prompt + model
           ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Router                            │
│  (OpenRouter API · model slots · retry with backoff       │
│   · asyncio.Semaphore to serialize concurrent calls)      │
└──────────────────────┬────────────────────────────────┘
                       │ tool calls
                       ▼
┌─────────────────────────────────────────────────────────┐
│               Tools (ProcessPoolExecutor)                  │
│  shell · files · brave_search · web_scrape · cron_mgr     │
│  Blocking tools (shell, web_scrape) run in process pool   │
└─────────────────────────────────────────────────────────┘
```

### Concurrency Model

The application uses a **single async event loop** with a **ProcessPoolExecutor** for CPU/IO-bound tool operations:

- **asyncio event loop** — drives Telegram polling, cron scheduling, and model API calls
- **ProcessPoolExecutor** (workers = 4) — shell commands and web scraping run in separate processes to prevent event loop starvation
- **asyncio.Semaphore** (permits = 1) — serializes model router calls so user messages and cron executions don't race on shared API state
- **Message queue** — when the bot is busy processing a message, incoming messages are queued and processed sequentially. A "thinking..." chat action is sent as backpressure signal
- **Health heartbeat** — main loop touches a heartbeat file every 30s; watchdog checks file freshness, not just process liveness

---

## Components

### 1. Telegram Bot

**Library:** python-telegram-bot (async)
**Auth:** Owner-only via `owner_id` in config. All messages from non-owner are silently ignored.

**Commands:**
- `/cron list` — list all crons with ID, name, schedule, next run, model, status
- `/cron remove <id>` — delete a cron
- `/cron pause <id>` — pause without deleting
- `/cron resume <id>` — resume paused cron
- `/cron info <id>` — details + last run result + recent failures
- `/config show` — show current runtime config
- `/config model <name>` — override default model for this session
- `/config reset` — reset overrides to config.yaml defaults
- `/status` — uptime, memory usage, DB size, active crons, last error
- `/search <query>` — full-text search over conversation history via FTS5
- `/forget` — start a new conversation (old one stays in DB, just starts fresh context)
- `/model <name>` — shortcut for `/config model <name>`

**Regular messages:** Stored in DB → sliding window context assembled → model router → response stored → sent back.

**Voice messages:** Downloaded → sent to Groq Whisper for transcription → transcribed text processed as a normal message. If Groq is not configured, reply with "Voice messages require Groq API key."

**Message backpressure:** When the bot is already processing a message, new messages are queued. A typing indicator (`chat_action = typing`) is sent immediately to signal the bot is busy. Messages are processed sequentially in FIFO order.

**Cron result delivery:**
- Fire-and-forget: result appears as a normal message from the bot
- On failure: always notify with error details (warning prefix)
- Cron outputs do NOT enter conversation memory
- If user replies to a cron result, the original cron output is included as one-off context for that turn only

### 2. Model Router

**Backend:** OpenRouter API (https://openrouter.ai/api/v1/chat/completions)

**Model slots (in config.yaml):**
- `default` — used for normal chat (e.g., `google/gemini-2.0-flash`)
- `smart` — used when user says `/model smart` or for complex tasks (e.g., `anthropic/claude-sonnet-4`)
- `cron_default` — default model for cron jobs (e.g., `google/gemini-2.0-flash`)

**Per-cron model:** Each cron can specify a model. Falls back to `cron_default`, then `default`.

**Tool-use:** Implements OpenAI-compatible function calling format via OpenRouter. Tools are registered as JSON schemas. Model responses with `tool_calls` are executed and results fed back in a loop until the model produces a final text response.

**Max tool iterations:** Configurable (default 20) to prevent runaway loops.

**Retry with exponential backoff:** All OpenRouter API calls use exponential backoff (base 1s, max 30s, 3 retries) for transient failures (429, 500, 502, 503, 504). Non-retryable errors (400, 401, 403) fail immediately.

**Serialization:** An `asyncio.Semaphore(1)` ensures only one model call runs at a time, preventing races between user messages and concurrent cron executions.

### 3. Context Manager (Sliding Window)

A simple, reliable context strategy that stores all messages and assembles a sliding window for each model call. Designed as a clean interface so LCM can replace it later without touching other components.

**Database:** SQLite at `~/.spare-paw/spare-paw.db`

#### Schema

```sql
-- Schema version tracking
PRAGMA user_version = 1;

-- Every message ever sent or received, verbatim
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,  -- ISO timestamp
    metadata TEXT  -- JSON: tool_call_id, tool_calls, model used, etc.
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);

-- Conversations
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_message_at TEXT,
    metadata TEXT  -- JSON
);

-- Cron jobs
CREATE TABLE cron_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,  -- cron expression
    prompt TEXT NOT NULL,
    model TEXT,  -- NULL = use cron_default
    tools_allowed TEXT,  -- JSON list of tool names, NULL = all
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_run_at TEXT,
    last_result TEXT,
    last_error TEXT,
    metadata TEXT  -- JSON
);

-- FTS5 for full-text search with proper sync triggers
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content=messages, content_rowid=rowid);

CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
```

#### Context Assembly (each turn)

1. Fetch the last `max_messages` (default 64) messages for the active conversation, ordered by `created_at`
2. Walk backwards from newest, accumulating token counts
3. Stop when `token_budget * safety_margin` is exceeded — drop remaining oldest messages
4. Prepend system prompt (with `{current_time}` interpolated)
5. Return as OpenAI-format message list

**Token counting:** tiktoken with `cl100k_base` encoding as a rough estimator. A configurable `safety_margin` (default 0.85) accounts for tokenizer drift across non-OpenAI models.

#### Configuration

```yaml
context:
  max_messages: 64        # sliding window size
  token_budget: 120000    # max tokens for context
  safety_margin: 0.85     # budget multiplier for tokenizer safety
```

#### Future: LCM Replacement

The context manager exposes a simple interface:
- `ingest(conversation_id, role, content, metadata)` → stores message
- `assemble(conversation_id)` → returns message list for model
- `search(query)` → FTS5 search

LCM can replace this module by implementing the same interface with DAG-based summarization, compaction, and deep recall. No other components need to change.

### 4. Tools

All tools are exposed to the LLM as callable functions (OpenAI function-calling format). Blocking tools run in a `ProcessPoolExecutor` to avoid starving the event loop.

#### shell
Execute a shell command in Termux.
- Parameters: `command` (string), `timeout` (int, default 30s)
- Returns: stdout, stderr, exit code
- Safety: configurable timeout, output truncated at 10K chars
- **Execution:** Runs in ProcessPoolExecutor

#### files
Read, write, list files on the phone.
- Actions: `read`, `write`, `append`, `list`, `delete`, `exists`
- Parameters: `path`, `content` (for write/append)
- Safety: restricted to `allowed_paths` in config (path traversal prevention via `os.path.realpath`)
- Returns: file content or directory listing or success/error

#### brave_search
Search the web using Brave Search API.
- Parameters: `query` (string), `count` (int, default 5)
- Returns: list of {title, url, description}
- Requires: Brave Search API key (free tier: 2000/month)
- If not configured: returns error suggesting configuration

#### web_scrape
Fetch and extract content from a specific URL.
- Parameters: `url` (string), `selector` (optional CSS selector)
- Returns: extracted text content (truncated at 20K chars)
- Uses: aiohttp + BeautifulSoup
- Timeout: 15s
- **Execution:** Runs in ProcessPoolExecutor (BeautifulSoup parsing is CPU-bound)

#### cron_create
Create a new scheduled task.
- Parameters: `name`, `schedule` (cron expression), `prompt`, `model` (optional), `tools_allowed` (optional list)
- Returns: cron ID and next run time
- Called by the LLM when user describes a scheduled task in natural language

#### cron_delete
Delete a scheduled task.
- Parameters: `cron_id`
- Returns: success/error

#### cron_list
List all scheduled tasks.
- Returns: list of {id, name, schedule, next_run, model, enabled, last_run_at}

### 5. Cron Scheduler

**Library:** APScheduler (AsyncIOScheduler)
**Persistence:** SQLite (cron_jobs table in spare-paw.db)
**Startup:** Loads all enabled crons from DB and schedules them.

**Execution flow:**
1. Cron fires at scheduled time
2. Scheduler acquires the model semaphore
3. Calls model router with the cron's prompt + allowed tools + specified model
4. Model executes (may call tools in a loop)
5. Final text response sent to owner via Telegram `bot.send_message(owner_id, result)`
6. On failure: send error notification with warning prefix
7. Update `last_run_at`, `last_result`, `last_error` in DB
8. Release semaphore

**Cron results are NOT stored in conversation memory.**
**If user replies to a cron result:** include original cron output as one-off context for that turn.

### 6. Voice Message Support

**Library:** Groq API (Whisper)
**Flow:**
1. User sends voice note on Telegram
2. Bot downloads the .ogg file
3. Sends to Groq Whisper endpoint for transcription
4. Transcribed text is processed as a normal message
5. Bot replies with transcription prefix, then the response

**If Groq not configured:** Reply with "Voice messages require a Groq API key in config."

### 7. Process Management

**Termux setup:**
- `termux-wake-lock` to prevent Android killing the process
- Main process: `python -m spare_paw gateway`
- Watchdog: bash script that monitors heartbeat file freshness (not just PID liveness)

**Logging:**
- Python `logging` module
- File handler with rotation (10MB max, 3 backups)
- Log location: `~/.spare-paw/logs/`

**Health heartbeat:**
- Main event loop touches `~/.spare-paw/heartbeat` every 30s
- Watchdog checks: if heartbeat file is older than 90s, kill and restart
- Catches event loop starvation and deadlocks, not just crashes

**Health command:**
- `/status` command shows: uptime, RAM usage, DB size, active crons, last error, current model config

### 8. Setup / Onboarding

**Command:** `python -m spare_paw setup`

Interactive wizard that:
1. Creates `~/.spare-paw/` directory structure
2. Generates `config.yaml` from template
3. Prompts for required API keys (OpenRouter, Telegram bot token)
4. Prompts for optional API keys (Brave Search, Groq)
5. Prompts for Telegram owner ID
6. Validates all keys by making test API calls
7. Initializes SQLite database
8. Prints next steps (how to start the gateway)

---

## Config File

Location: `~/.spare-paw/config.yaml`

```yaml
telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  owner_id: 123456789

openrouter:
  api_key: "YOUR_OPENROUTER_API_KEY"

models:
  default: "google/gemini-2.0-flash"
  smart: "anthropic/claude-sonnet-4"
  cron_default: "google/gemini-2.0-flash"

brave:
  api_key: ""  # optional, free tier (2000 queries/month)

groq:
  api_key: ""  # optional, for voice transcription

context:
  max_messages: 64        # sliding window size
  token_budget: 120000    # max tokens for context
  safety_margin: 0.85     # budget multiplier for tokenizer safety

tools:
  shell:
    enabled: true
    timeout_seconds: 30
    max_output_chars: 10000
  files:
    enabled: true
    allowed_paths:
      - "/sdcard"
      - "/data/data/com.termux/files/home"
  brave_search:
    enabled: true
    max_results: 5
  web_scrape:
    enabled: true
    timeout_seconds: 15
    max_content_chars: 20000
  cron:
    enabled: true

agent:
  max_tool_iterations: 20
  system_prompt: |
    You are a personal AI assistant running 24/7 on an Android phone.
    You have access to the local filesystem, shell, web search, and web scraping.
    You can manage scheduled tasks (crons) for the user.
    Be concise. The user is on Telegram, likely on mobile.
    Current time: {current_time}
    Device: Android (Termux)

logging:
  level: "INFO"
  max_bytes: 10485760  # 10MB
  backup_count: 3
```

---

## Project Structure

```
spare-paw/
├── pyproject.toml
├── SPEC.md
├── scripts/
│   ├── watchdog.sh          # Heartbeat-aware restart script
│   └── install-termux.sh    # Termux dependency installer
├── src/
│   └── spare_paw/
│       ├── __init__.py
│       ├── __main__.py      # Entry point: setup / gateway
│       ├── config.py        # Config loading (YAML + runtime overrides)
│       ├── db.py            # SQLite connection, schema, migrations
│       ├── context.py       # Sliding window context assembly + FTS5 search
│       ├── gateway.py       # Main async loop: bot + scheduler + heartbeat
│       ├── setup_wizard.py  # Interactive onboarding wizard
│       │
│       ├── bot/
│       │   ├── __init__.py
│       │   ├── handler.py   # Message handler with queue + backpressure
│       │   ├── commands.py  # /cron, /config, /status, /search, /forget, /model
│       │   └── voice.py     # Voice message transcription (Groq Whisper)
│       │
│       ├── router/
│       │   ├── __init__.py
│       │   ├── openrouter.py  # OpenRouter API client with retry/backoff
│       │   └── tool_loop.py   # Tool-use execution loop
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py    # Tool registration & JSON schema generation
│       │   ├── shell.py       # Shell command execution (ProcessPoolExecutor)
│       │   ├── files.py       # File read/write/list (path-restricted)
│       │   ├── brave_search.py # Brave Search API
│       │   ├── web_scrape.py  # URL fetching + BeautifulSoup (ProcessPoolExecutor)
│       │   └── cron_tools.py  # cron_create, cron_delete, cron_list
│       │
│       └── cron/
│           ├── __init__.py
│           ├── scheduler.py   # APScheduler setup & job management
│           └── executor.py    # Cron job execution (semaphore-gated)
│
└── config.example.yaml
```

---

## Dependencies

```
python-telegram-bot>=21.0        # Telegram bot (async)
aiohttp>=3.9                     # Async HTTP (OpenRouter, Brave, Groq)
apscheduler>=3.10,<4             # Cron scheduling
beautifulsoup4>=4.12             # Web scraping (HTML parsing)
tiktoken>=0.7                    # Token counting (approximate)
pyyaml>=6.0                      # Config file parsing
aiosqlite>=0.20                  # Async SQLite
```

---

## Termux Installation

```bash
# Install base packages
pkg update && pkg upgrade
pkg install python python-pip git

# Install spare-paw
git clone <repo>
cd spare-paw
pip install --break-system-packages -e .

# Setup (interactive wizard)
python -m spare_paw setup

# Run
python -m spare_paw gateway

# With watchdog (recommended)
bash scripts/watchdog.sh
```

---

## Key Design Decisions

1. **Python over Node.js** — Better Termux support, native SQLite, stronger ecosystem for web scraping and Telegram bots.

2. **OpenRouter API** — Any model, no CLI overhead, configurable per-slot and per-cron.

3. **Sliding window context (v1) with LCM interface** — Ship fast with simple last-N-messages context. The context manager exposes `ingest/assemble/search` — LCM replaces this module later without touching bot, router, or tools.

4. **ProcessPoolExecutor for blocking tools** — Shell commands and web scraping run in separate processes. The async event loop stays responsive even during long-running tool calls.

5. **Semaphore-serialized model calls** — Prevents races between user messages and concurrent cron executions hitting the model API simultaneously. One model call at a time.

6. **Message queue with backpressure** — Incoming Telegram messages queue while the bot is processing. Typing indicator shows the bot is busy. No message loss, no cascading delays.

7. **Heartbeat-based watchdog** — Watchdog checks file freshness (not just PID). Catches event loop starvation and deadlocks that a simple process monitor would miss.

8. **FTS5 with sync triggers** — Full-text search stays current via AFTER INSERT/DELETE/UPDATE triggers. No stale index.

9. **Exponential backoff on all external APIs** — Flaky phone Wi-Fi is a given. All HTTP calls (OpenRouter, Brave, Groq) retry with backoff for transient errors.

10. **Crons separate from conversation** — Cron outputs don't pollute context. Replies to cron results get one-off context inclusion.

11. **Schema versioning** — `PRAGMA user_version` tracks schema version. Future migrations check version on startup and apply incremental changes.

12. **Token safety margin** — tiktoken estimates are approximate for non-OpenAI models. An 0.85 multiplier on the budget prevents context overflows from tokenizer drift.
