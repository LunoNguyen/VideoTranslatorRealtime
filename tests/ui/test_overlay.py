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


# -- streaming / speaker colors ------------------------------------------------
def test_partial_italic_then_final_bold_same_id(qtbot):
    from ui.overlay import PALETTE

    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("xin", sub_id=1, final=False, speaker=0)
    assert "italic" in ov.vi_label.styleSheet()
    assert "bold" not in ov.vi_label.styleSheet().replace("font-weight: 600", "")
    ov.show_subtitle("xin chào", sub_id=1, final=True, speaker=0)
    assert "italic" not in ov.vi_label.styleSheet()
    assert "bold" in ov.vi_label.styleSheet()
    assert PALETTE[0] in ov.vi_label.styleSheet()


def test_partial_replaced_by_final_same_id(qtbot):
    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("xin", sub_id=1, final=False)
    ov.show_subtitle("xin chào", sub_id=1, final=True)
    assert ov.vi_label.text() == "xin chào"
    assert ov.prev_label.text() == ""


def test_overlay_ignores_partial_after_final(qtbot):
    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("xin chào", sub_id=1, final=True)
    ov.show_subtitle("xin", sub_id=1, final=False)
    assert ov.vi_label.text() == "xin chào"
    assert "italic" not in ov.vi_label.styleSheet()


def test_new_id_moves_final_to_prev_line(qtbot):
    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("một", sub_id=1, final=True)
    ov.show_subtitle("hai", sub_id=2, final=False)
    assert ov.prev_label.text() == "một" and ov.vi_label.text() == "hai"
    assert not ov.prev_label.isHidden()


def test_speaker_color_applied(qtbot):
    from ui.overlay import PALETTE

    ov = make(qtbot)
    ov.show_subtitle("a", sub_id=1, final=True, speaker=1)
    assert PALETTE[1] in ov.vi_label.styleSheet()


def test_none_speaker_white(qtbot):
    ov = make(qtbot)
    ov.show_subtitle("a", sub_id=1, final=True, speaker=None)
    assert "#FFFFFF" in ov.vi_label.styleSheet()


def test_provisional_without_final_is_cleared_by_timer(qtbot):
    ov = make(qtbot, hold=0.2)
    ov.show_subtitle("một", sub_id=1, final=True)
    ov.show_subtitle("hai", sub_id=2, final=False)
    qtbot.wait(350)
    assert ov.vi_label.text() == "" and ov.prev_label.text() == ""
    assert not ov.isVisible()


def test_timer_restarts_on_each_update(qtbot):
    ov = make(qtbot, hold=0.3)
    ov.show_subtitle("a", sub_id=1, final=False)
    qtbot.wait(200)
    ov.show_subtitle("ab", sub_id=1, final=False)
    qtbot.wait(200)
    assert ov.vi_label.text() == "ab"
    qtbot.wait(250)
    assert ov.vi_label.text() == ""


def test_source_shown_for_current_line_only(qtbot):
    ov = make(qtbot, hold=1.0, show_source=True)
    ov.show_subtitle("một", "one", sub_id=1, final=True)
    ov.show_subtitle("hai", "two", sub_id=2, final=True)
    assert ov.src_label.text() == "two" and ov.prev_label.text() == "một"


def test_prev_line_keeps_speaker_color(qtbot):
    from ui.overlay import PALETTE

    ov = make(qtbot, hold=1.0)
    ov.show_subtitle("một", sub_id=1, final=True, speaker=2)
    ov.show_subtitle("hai", sub_id=2, final=True, speaker=0)
    assert ov.prev_label.text() == "một"
    assert PALETTE[2] in ov.prev_label.styleSheet()
    assert "font-size: 18px" in ov.prev_label.styleSheet()
    assert ov.prev_label.graphicsEffect().opacity() == 0.6


def test_reset_lines_prevents_id_collision_after_restart(qtbot):
    ov = make(qtbot, hold=5.0)
    ov.show_subtitle("cũ", sub_id=1, final=True)
    ov.show_subtitle("cũ2", sub_id=2, final=True)
    ov.reset_lines()
    assert ov.prev_label.text() == "" and ov.vi_label.text() == ""
    ov.show_subtitle("mới", sub_id=1, final=False)  # id 1 restarts; must not be ignored
    assert ov.vi_label.text() == "mới"
    ov.reset_lines()
    ov.show_subtitle("mới2", sub_id=2, final=False)
    assert ov.vi_label.text() == "mới2"
