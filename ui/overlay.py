"""Frameless always-on-top subtitle overlay."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget

CONNECTING_TEXT = "Đang kết nối…"
_VI_STYLE = "color: white; font-size: 28px; font-weight: 600;"
_PREV_STYLE = "color: {color}; font-size: 18px;"
PALETTE = ["#FFD54F", "#4FC3F7", "#81C784", "#FF8A65", "#BA68C8", "#F06292"]
_DEFAULT_COLOR = "#FFFFFF"
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

        self._cur_id: int | None = None
        self._cur_final = False
        self._prev_id: int | None = None
        self._cur_color = _DEFAULT_COLOR

        self.prev_label = self._label(_PREV_STYLE.format(color=_DEFAULT_COLOR))
        self.prev_label.hide()
        fx = QGraphicsOpacityEffect(self.prev_label)
        fx.setOpacity(0.6)
        self.prev_label.setGraphicsEffect(fx)
        self.vi_label = self._label(_VI_STYLE)
        self._vi_fx = QGraphicsOpacityEffect(self.vi_label)
        self._vi_fx.setOpacity(1.0)
        self.vi_label.setGraphicsEffect(self._vi_fx)
        self.src_label = self._label(_SRC_STYLE)
        self.src_label.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(4)
        lay.addWidget(self.prev_label)
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

    def show_subtitle(
        self,
        vi_text: str,
        src_text: str | None = None,
        *,
        sub_id: int | None = None,
        final: bool = True,
        speaker: int | None = None,
    ) -> None:
        self._connecting = False
        same = sub_id is not None and sub_id == self._cur_id
        if sub_id is not None and not final and (
            (same and self._cur_final) or sub_id == self._prev_id
        ):
            return  # a late partial never overrides a final line
        if not same:
            if self._cur_final and self.vi_label.text():
                self.prev_label.setText(self.vi_label.text())
                self.prev_label.setStyleSheet(
                    _PREV_STYLE.format(color=self._cur_color) + " background: transparent;"
                )
                self.prev_label.show()
                self._prev_id = self._cur_id
            else:  # stale provisional line is dropped, never promoted
                self.prev_label.clear()
                self.prev_label.hide()
        self._cur_id, self._cur_final = sub_id, bool(final)
        self.vi_label.setText(vi_text)
        self._style_current(final, speaker)
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

    def _style_current(self, final: bool, speaker: int | None) -> None:
        if isinstance(speaker, int) and not isinstance(speaker, bool) and speaker >= 0:
            color = PALETTE[speaker % len(PALETTE)]
        else:
            color = _DEFAULT_COLOR
        self._cur_color = color
        if final:
            style = f"color: {color}; font-size: 28px; font-weight: bold;"
        else:
            style = f"color: {color}; font-size: 28px; font-weight: 400; font-style: italic;"
        self.vi_label.setStyleSheet(style + " background: transparent;")
        self._vi_fx.setOpacity(1.0 if final else 0.7)

    def set_connecting(self, connecting: bool) -> None:
        self._connecting = connecting
        self._timer.stop()
        self.src_label.clear()
        self.src_label.hide()
        self._reset_lines()
        self.vi_label.setText(CONNECTING_TEXT if connecting else "")
        if connecting:
            self._style_current(True, None)
        self._refresh_visibility()

    def clear(self) -> None:
        if self._connecting:
            return
        self.vi_label.clear()
        self.src_label.clear()
        self.src_label.hide()
        self._reset_lines()
        self._refresh_visibility()

    def reset_lines(self) -> None:
        """Forget the current/previous line ids and text (ids restart after a pipeline restart)."""
        self._timer.stop()
        self.src_label.clear()
        self.src_label.hide()
        if not self._connecting:
            self.vi_label.clear()
        self._reset_lines()
        self._refresh_visibility()

    def _reset_lines(self) -> None:
        self.prev_label.clear()
        self.prev_label.hide()
        self._cur_id, self._cur_final, self._prev_id = None, False, None
        self._cur_color = _DEFAULT_COLOR

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
