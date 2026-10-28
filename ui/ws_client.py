"""Qt WebSocket client for the local backend (reconnects every 3 s)."""
from __future__ import annotations

import json

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket

RECONNECT_MS = 3000


class WSClient(QObject):
    subtitle = Signal(dict)
    apps = Signal(list)
    status = Signal(dict)
    config = Signal(dict)
    connected = Signal(bool)

    def __init__(self, url: str = "ws://127.0.0.1:8765/ws", autoconnect: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._url = QUrl(url)
        self._ws = QWebSocket()
        self._ws.connected.connect(self._on_connected)
        self._ws.disconnected.connect(self._on_disconnected)
        self._ws.textMessageReceived.connect(self.handle_text)
        self._timer = QTimer(self)
        self._timer.setInterval(RECONNECT_MS)
        self._timer.timeout.connect(self._open)
        self._is_connected = False
        self._started = False
        if autoconnect:
            self.start()

    def start(self) -> None:
        self._started = True
        self._open()
        self._timer.start()

    def stop(self) -> None:
        self._started = False
        self._timer.stop()
        self._ws.close()

    def is_connected(self) -> bool:
        return self._is_connected

    def _open(self) -> None:
        if self._started and not self._is_connected:
            self._ws.abort()
            self._ws.open(self._url)

    def _on_connected(self) -> None:
        self._is_connected = True
        self.connected.emit(True)
        self.send("status")
        self.send("list_apps")

    def _on_disconnected(self) -> None:
        was = self._is_connected
        self._is_connected = False
        if was:
            self.connected.emit(False)

    def send(self, type: str, **payload) -> bool:  # noqa: A002 - protocol field name
        if not self._is_connected:
            return False
        self._ws.sendTextMessage(json.dumps({"type": type, **payload}))
        return True

    def handle_text(self, raw: str) -> None:
        """Dispatch one raw server message to the matching signal."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "subtitle":
            self.subtitle.emit(msg)
        elif kind == "apps" and isinstance(msg.get("apps"), list):
            self.apps.emit(msg["apps"])
        elif kind == "status":
            self.status.emit(msg)
        elif kind == "config" and isinstance(msg.get("config"), dict):
            self.config.emit(msg["config"])
