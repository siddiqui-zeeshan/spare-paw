"""Bottom status bar widget."""

from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    """Shows connection, model, message count, tool count."""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._connection = "local"
        self._url = "local"
        self._model = "unknown"
        self._msg_count = 0
        self._tool_count = 0

    def set_state(
        self,
        connection: str,
        url: str,
        model: str,
        msg_count: int,
        tool_count: int,
    ) -> None:
        self._connection = connection
        self._url = url
        self._model = model
        self._msg_count = msg_count
        self._tool_count = tool_count
        self.update(self.render_text())

    def render_text(self) -> str:
        color_map = {
            "connected": "[green]◉[/green]",
            "reconnecting": "[yellow]◉[/yellow]",
            "disconnected": "[red]◉[/red]",
            "local": "[cyan]◉[/cyan]",
        }
        dot = color_map.get(self._connection, "[red]◉[/red]")
        parts = [
            f"{dot} {self._url}",
            self._model,
            f"{self._msg_count} msgs",
            f"↑{self._tool_count} tools",
        ]
        return "  │  ".join(parts)
