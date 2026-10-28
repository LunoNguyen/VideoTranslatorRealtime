"""Frameless always-on-top subtitle overlay."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

CONNECTING_TEXT = "Đang kết nối…"
_VI_STYLE = "color: white; font-size: 28px; font-weight: 600;"
_SRC_STYLE = "color: #d0d0d0; font-size: 16px;"


class Overlay(QWidget):
    """Draggable subtitle bar; position is remembered in QSettings (no backend config key exists)."""

    def __init__(self, settings_group: str = "overlay") -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._group = settings_group
        self._hold_ms = 4000
        self._show_source = False
        self._connecting = False
        self._drag: QPoint | None = None
        self._paused_ms: int | None = None

        self.vi_label = self._label(_VI_STYLE)
        self.src_label = self._label(_SRC_STYLE)
        self.src_label.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(4)
        lay.addWidget(self.vi_label)
        lay.addWidget(self.src_label)
        self.setMinimumWidth(480)
        self.setMaximumWidth(1200)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.clear)
        self._restore_pos()

    @staticmethod
    def _label(style: str) -> QLabel:
        lb = QLabel("")
        lb.setStyleSheet(style + " background: transparent;")
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.setWordWrap(True)
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        return lb

    # -- API ---------------------------------------------------------------
    def set_hold_seconds(self, seconds: float) -> None:
        self._hold_ms = max(0, int(float(seconds) * 1000))

    def set_show_source(self, show: bool) -> None:
        self._show_source = bool(show)
        if not self._show_source:
            self.src_label.clear()
            self.src_label.hide()

    def show_subtitle(self, vi_text: str, src_text: str | None = None) -> None:
        self._connecting = False
        self.vi_label.setText(vi_text)
        if self._show_source and src_text:
            self.src_label.setText(src_text)
            self.src_label.show()
        else:
            self.src_label.clear()
            self.src_label.hide()
        if self._drag is not None:  # dragging: keep paused, restart with the full hold on release
            self._timer.stop()
            self._paused_ms = self._hold_ms
        else:
            self._timer.start(self._hold_ms)
        self._refresh_visibility()

    def set_connecting(self, connecting: bool) -> None:
        self._connecting = connecting
        self._timer.stop()
        self.src_label.clear()
        self.src_label.hide()
        self.vi_label.setText(CONNECTING_TEXT if connecting else "")
        self._refresh_visibility()

    def clear(self) -> None:
        if self._connecting:
            return
        self.vi_label.clear()
        self.src_label.clear()
        self.src_label.hide()
        self._refresh_visibility()

    # -- internals -----------------------------------------------------------
    def _refresh_visibility(self) -> None:
        """Hide the bar entirely when there is no text so it never blocks the video."""
        if self.vi_label.text():
            self.adjustSize()
            if not self.isVisible():
                self.show()
        elif self.isVisible():
            self.hide()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(0, 0, 0, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)

    @staticmethod
    def _settings() -> QSettings:
        return QSettings("VideoTranslatorRealtime", "ui")

    def _restore_pos(self) -> None:
        try:
            pos = self._settings().value(f"{self._group}/pos")
            if isinstance(pos, QPoint) and any(
                s.availableGeometry().contains(pos) for s in QGuiApplication.screens()
            ):
                self.move(pos)
                return
        except Exception:  # noqa: BLE001 - never fail on settings
            pass
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.move(g.center().x() - 300, g.bottom() - 160)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            if self._timer.isActive():  # pause the hold timer while dragging
                self._paused_ms = max(self._timer.remainingTime(), 0)
                self._timer.stop()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag = None
            if self._paused_ms is not None:
                self._timer.start(self._paused_ms)
                self._paused_ms = None
            self._settings().setValue(f"{self._group}/pos", self.pos())
