"""Status mapping and MainWindow wiring."""
from __future__ import annotations

from ui.main_window import STATUS_TEXT, MainWindow, describe_status
from ui.overlay import Overlay
from ui.ws_client import WSClient


def test_status_mapping_known_unknown_quiet():
    text, style = describe_status({"level": "error", "code": "model_missing", "message": "Thiếu X. Chạy scripts/download_models.py"})
    assert "Chạy scripts/download_models.py" in text and style == "error"
    assert describe_status({"level": "warn", "code": "dropped_segment", "message": "x"})[1] == "quiet"
    assert describe_status({"level": "error", "code": "zzz", "message": "raw msg"}) == ("raw msg", "error")
    assert len(STATUS_TEXT) == 19


def _clear_group(group: str) -> None:
    from PySide6.QtCore import QSettings

    QSettings("VideoTranslatorRealtime", "ui").remove(group)


def test_window_sends_and_applies(qtbot):
    _clear_group("test_win")
    try:
        c = WSClient(autoconnect=False)
        sent = []
        c.send = lambda kind, **p: sent.append((kind, p)) or True
        ov = Overlay(settings_group="test_win")
        qtbot.addWidget(ov)
        w = MainWindow(c, ov, settings_group="test_win")
        qtbot.addWidget(w)
        c.apps.emit([{"exe": "chrome.exe", "display_name": "chrome", "active": True, "allowed": False}])
        assert w.table.rowCount() == 1
        w.table.cellWidget(0, 2).findChild(type(w.tts_check)).setChecked(True)
        assert ("set_allowed", {"exe": "chrome.exe", "allowed": True}) in sent
        w.model_combo.setCurrentIndex(1)
        w.model_combo.activated.emit(1)
        assert ("set_models", {"asr_model": "medium"}) in sent
        n = len(sent)
        c.config.emit({"langs": {"src": "ja"}, "tts": {"enabled": True, "volume": 0.5}, "overlay": {"show_source": True, "hold_seconds": 2.0}})
        assert len(sent) == n and w.src_combo.currentData() == "ja" and w.volume.value() == 50
        c.status.emit({"level": "warn", "code": "cuda_unavailable", "message": "x"})
        assert "CUDA" in w.status_label.text()
    finally:
        _clear_group("test_win")


def test_custom_model_from_config_gets_extra_item(qtbot):
    _clear_group("test_custom")
    try:
        c = WSClient(autoconnect=False)
        sent = []
        c.send = lambda kind, **p: sent.append((kind, p)) or True
        w = MainWindow(c, None, settings_group="test_custom")
        qtbot.addWidget(w)
        c.config.emit({"asr": {"model": "large-v3"}})
        c.config.emit({"asr": {"model": "large-v3"}})  # no duplicate item
        assert w.model_combo.currentData() == "large-v3"
        assert w.model_combo.currentText() == "large-v3 (tùy chỉnh)"
        assert w.model_combo.count() == 3
        assert sent == []
        c.config.emit({"asr": {"model": "small"}})
        assert w.model_combo.currentData() == "small" and sent == []
    finally:
        _clear_group("test_custom")


def test_show_source_survives_config(qtbot):
    _clear_group("test_fix")
    try:
        c = WSClient(autoconnect=False)
        c.send = lambda *a, **k: True
        ov = Overlay(settings_group="test_fix")
        qtbot.addWidget(ov)
        w = MainWindow(c, ov, settings_group="test_fix")
        qtbot.addWidget(w)
        w.source_check.setChecked(True)
        c.config.emit({"overlay": {"show_source": False, "hold_seconds": 2.0}})
        assert w.source_check.isChecked()
        ov.show_subtitle("vi", "src")
        assert ov.src_label.text() == "src"
        ov2 = Overlay(settings_group="test_fix")
        qtbot.addWidget(ov2)
        w2 = MainWindow(c, ov2, settings_group="test_fix")
        qtbot.addWidget(w2)
        assert w2.source_check.isChecked()  # persisted
    finally:
        _clear_group("test_fix")


def test_self_tree_status_names_app():
    text, style = describe_status({"level": "warn", "code": "self_tree",
                                   "message": "Không thể nghe cmd.exe vì nó là tiến trình cha của ứng dụng này"})
    assert text == "Không thể nghe cmd.exe vì nó là tiến trình cha của ứng dụng này" and style == "warn"


def test_stream_and_speaker_checkboxes_send(qtbot):
    c = WSClient(autoconnect=False)
    sent = []
    c.send = lambda kind, **p: sent.append((kind, p)) or True
    w = MainWindow(c, None, settings_group="test_chk")
    qtbot.addWidget(w)
    assert w.stream_check.text() == "Dịch tức thì" and w.speaker_check.text() == "Tô màu người nói"
    w.stream_check.setChecked(True)
    w.speaker_check.setChecked(True)
    assert ("set_streaming", {"enabled": True}) in sent
    assert ("set_speaker_colors", {"enabled": True}) in sent
    _clear_group("test_chk")


def test_apply_config_sets_new_checkboxes(qtbot):
    c = WSClient(autoconnect=False)
    sent = []
    c.send = lambda kind, **p: sent.append((kind, p)) or True
    w = MainWindow(c, None, settings_group="test_chk2")
    qtbot.addWidget(w)
    c.config.emit({"streaming": {"enabled": True}, "speaker": {"enabled": True}})
    assert w.stream_check.isChecked() and w.speaker_check.isChecked() and sent == []
    c.config.emit({"streaming": {"enabled": False}, "speaker": {"enabled": False}})
    assert not w.stream_check.isChecked() and not w.speaker_check.isChecked() and sent == []
    _clear_group("test_chk2")


def test_new_status_codes_have_text():
    assert STATUS_TEXT["streaming_slow"] == "Máy không theo kịp, đã chuyển về dịch theo câu."
    assert STATUS_TEXT["speaker_unavailable"] == "Không dùng được tô màu người nói."
    text, _ = describe_status({"level": "warn", "code": "streaming_slow", "message": "x"})
    assert text == STATUS_TEXT["streaming_slow"]
    raw = "Không có model phân biệt giọng. Chạy scripts/download_models.py --only speaker"
    text, lvl = describe_status({"level": "warn", "code": "speaker_unavailable", "message": raw})
    assert text == raw and lvl == "warn"
    text, _ = describe_status({"level": "warn", "code": "speaker_unavailable", "message": ""})
    assert text == STATUS_TEXT["speaker_unavailable"]
