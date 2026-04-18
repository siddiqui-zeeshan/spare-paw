"""Centralized CSS and color tokens for the TUI.

Keeping constants here means CSS strings aren't scattered across widget
modules and colors can be changed in one place.
"""

from __future__ import annotations

# Token coalescing window for live streaming. Tokens arriving within this
# window are batched into a single widget append to keep render cost down.
STREAM_COALESCE_MS = 16

# Bubble label column width used to pad "You"/"spare-paw" vs timestamp.
# Computed at render time from terminal width; this is a minimum.
MIN_LABEL_COLUMN = 40

APP_CSS = """
Screen {
    background: $surface;
}

#chat-log {
    height: 1fr;
    padding: 1 0;
    background: $surface;
}

MessageView {
    layout: vertical;
    padding: 0 2;
    height: auto;
    width: 100%;
}

MessageView.user .header { color: $success; text-style: bold; }
MessageView.assistant .header { color: $accent; text-style: bold; }

/* Disable Markdown widget's internal scrollbar so ChatLog owns scrolling. */
Markdown {
    height: auto;
    overflow-x: hidden;
    overflow-y: hidden;
    scrollbar-size: 0 0;
}
MarkdownTable, MarkdownFence, MarkdownBlockQuote {
    overflow-x: hidden;
    scrollbar-size: 0 0;
}

ToolRow {
    padding: 0 0 0 2;
    height: auto;
}
ToolRow.running { color: $warning; }
ToolRow.success { color: $text-muted; }
ToolRow.error { color: $error; }
ToolRow.cancelled { color: $text-disabled; }

#composer {
    border-top: solid $primary-darken-2;
    padding: 0 1;
    height: auto;
    max-height: 10;
}

#status-bar {
    height: 1;
    background: $panel;
    color: $text-muted;
    padding: 0 1;
    dock: bottom;
}

.connection-green { color: $success; }
.connection-yellow { color: $warning; }
.connection-red { color: $error; }
"""
