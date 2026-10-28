"""Overlay behaviour tests (offscreen Qt)."""
from __future__ import annotations

from ui.overlay import Overlay


def make(qtbot, hold: float = 0.2, show_source: bool = False) -> Overlay:
    ov = Overlay(settings_group="test_overlay")
    qtbot.addWidget(ov)
    ov.set_hold_seconds(hold)
    ov.set_show_source(show_source)
    return ov


def test_subtitle_hides_after_hold(qtbot):
    ov = make(qtbot, hold=0.2)
    ov.show_subtitle("Xin chào", None)
    assert ov.vi_label.text() == "Xin chào"
    qtbot.wait(300)
    assert ov.vi_label.text() == ""


def test_new_subtitle_replaces_old(qtbot):
    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("một", None)
    ov.show_subtitle("hai", None)
    assert ov.vi_label.text() == "hai"


def test_source_line_hidden_by_default(qtbot):
    ov = make(qtbot)
    ov.show_subtitle("xin chào", "hello")
    assert ov.src_label.isHidden() or ov.src_label.text() == ""
    ov.set_show_source(True)
    ov.show_subtitle("xin chào", "hello")
    assert ov.src_label.text() == "hello"
    assert not ov.src_label.isHidden()


def test_connecting_indicator(qtbot):
    ov = make(qtbot)
    ov.set_connecting(True)
    assert ov.vi_label.text() == "Đang kết nối…"
    ov.set_connecting(False)
    assert ov.vi_label.text() == ""


def test_drag_pauses_hold_timer(qtbot):
    from PySide6.QtCore import QPoint, Qt

    ov = make(qtbot, hold=0.2)
    ov.show_subtitle("giữ", None)
    qtbot.mousePress(ov, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(350)
    assert ov.vi_label.text() == "giữ"
    qtbot.mouseRelease(ov, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(350)
    assert ov.vi_label.text() == ""


def test_subtitle_during_drag_gets_full_hold_after_release(qtbot):
    from PySide6.QtCore import QPoint, Qt

    ov = make(qtbot, hold=0.3)
    ov.show_subtitle("một", None)
    qtbot.mousePress(ov, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(250)
    ov.show_subtitle("hai", None)  # arrives mid-drag
    qtbot.wait(400)
    assert ov.vi_label.text() == "hai"  # still paused while dragging
    qtbot.mouseRelease(ov, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(150)
    assert ov.vi_label.text() == "hai"  # new hold time, not the stale remainder of "một"
    qtbot.wait(350)
    assert ov.vi_label.text() == ""
