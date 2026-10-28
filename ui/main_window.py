"""Settings window: app list, languages, TTS, model preset and status area."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QSlider,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ui.ws_client import WSClient

# code -> Vietnamese user message
STATUS_TEXT = {
    "loopback_unsupported": "Windows của bạn không hỗ trợ thu âm theo từng ứng dụng (cần Windows 10 2004 trở lên).",
    "model_missing": "Thiếu model dịch/nhận dạng giọng nói.",
    "cuda_oom": "Hết bộ nhớ card đồ họa (VRAM).",
    "cuda_unavailable": "Không dùng được CUDA, đã chuyển sang CPU (sẽ chậm hơn).",
    "app_closed": "Ứng dụng đã đóng.",
    "dropped_segment": "Bỏ qua một đoạn vì xử lý chưa kịp.",
    "models_loading": "Đang tải model, vui lòng đợi…",
    "models_ready": "Model đã sẵn sàng.",
    "model_load_failed": "Không tải được model.",
    "capture_failed": "Không thu được âm thanh của ứng dụng.",
    "bad_message": "Yêu cầu không hợp lệ.",
    "internal_error": "Lỗi nội bộ của dịch vụ.",
    "sessions_failed": "Không đọc được danh sách ứng dụng đang phát âm thanh.",
    "tts_error": "Lỗi đọc giọng nói.",
    "config_save_failed": "Không lưu được cấu hình.",
    "pipeline_error": "Lỗi trong quá trình xử lý âm thanh.",
    "self_tree": "Không thể nghe ứng dụng này vì nó là tiến trình cha của ứng dụng này.",
}
_QUIET = {"dropped_segment"}
_COLORS = {"error": "#c62828", "warn": "#b26a00", "info": "#2e7d32", "quiet": "#888888"}

SRC_CHOICES = [("Tự động", "auto"), ("English", "en"), ("日本語", "ja")]
MODEL_CHOICES = [("4GB VRAM (small)", "small"), ("6GB VRAM (medium)", "medium")]


def describe_status(msg: dict) -> tuple[str, str]:
    """Return (text, style key) for a status message; unknown codes show the raw message."""
    code, raw, level = msg.get("code", ""), str(msg.get("message", "")), msg.get("level", "info")
    if code not in STATUS_TEXT:
        return raw or code or "Không rõ trạng thái", "error" if level == "error" else level if level in _COLORS else "info"
    text = STATUS_TEXT[code]
    # backend message carries details/hints (app name, how to fix): append when present
    if raw and code in ("app_closed", "self_tree"):
        text = raw  # already names the app
    elif raw and code not in ("models_loading", "models_ready", "dropped_segment"):
        text = f"{text} {raw}"
    if code in _QUIET:
        return text, "quiet"
    return text, level if level in _COLORS else "info"


class MainWindow(QWidget):
    def __init__(self, client: WSClient, overlay=None, settings_group: str = "main") -> None:
        super().__init__()
        self.client, self.overlay = client, overlay
        self._group = settings_group
        self._updating = False  # suppress echo while applying server state
        self.setWindowTitle("Dịch video thời gian thực")
        self.resize(560, 560)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Ứng dụng", "Trạng thái", "Cho phép nghe"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.src_combo = QComboBox()
        for label, val in SRC_CHOICES:
            self.src_combo.addItem(label, val)
        self.tgt_combo = QComboBox()
        self.tgt_combo.addItem("Tiếng Việt", "vi")
        self.model_combo = QComboBox()
        for label, val in MODEL_CHOICES:
            self.model_combo.addItem(label, val)
        self.tts_check = QCheckBox("Đọc bản dịch (TTS)")
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(100)
        self.source_check = QCheckBox("Hiện câu gốc")
        self.conn_label = QLabel("Đang kết nối…")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(48)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        form = QFormLayout()
        form.addRow("Ngôn ngữ nguồn", self.src_combo)
        form.addRow("Ngôn ngữ đích", self.tgt_combo)
        form.addRow("Model", self.model_combo)
        form.addRow(self.tts_check)
        form.addRow("Âm lượng đọc", self.volume)
        form.addRow(self.source_check)
        top = QHBoxLayout()
        top.addWidget(QLabel("Chọn ứng dụng cần dịch âm thanh:"))
        top.addStretch(1)
        top.addWidget(self.conn_label)
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.table, 1)
        root.addLayout(form)
        root.addWidget(self.status_label)

        self.src_combo.activated.connect(self._send_langs)
        self.model_combo.activated.connect(self._send_model)
        self.tts_check.toggled.connect(lambda v: self._send("set_tts", enabled=v))
        self.volume.sliderReleased.connect(lambda: self._send("set_volume", value=self.volume.value() / 100))
        self.volume.actionTriggered.connect(self._volume_step)

        # show_source is a local preference (the backend never stores it): the saved value wins
        saved = QSettings("VideoTranslatorRealtime", "ui").value(f"{self._group}/show_source", False)
        on = str(saved).lower() in ("true", "1")
        self.source_check.setChecked(on)
        if overlay is not None:
            overlay.set_show_source(on)
        self.source_check.toggled.connect(self._on_source_toggled)

        client.apps.connect(self.set_apps)
        client.status.connect(self.show_status)
        client.config.connect(self.apply_config)
        client.connected.connect(self.set_connected)

    # -- outgoing --------------------------------------------------------------
    def _send(self, kind: str, **payload) -> None:
        if not self._updating:
            self.client.send(kind, **payload)

    def _send_langs(self) -> None:
        self._send("set_langs", src=self.src_combo.currentData(), tgt=self.tgt_combo.currentData())

    def _send_model(self) -> None:
        self._send("set_models", asr_model=self.model_combo.currentData())

    def _volume_step(self, _action) -> None:
        if not self.volume.isSliderDown():  # keyboard / wheel / page step
            self._send("set_volume", value=self.volume.value() / 100)

    def _on_source_toggled(self, on: bool) -> None:
        QSettings("VideoTranslatorRealtime", "ui").setValue(f"{self._group}/show_source", on)
        if self.overlay is not None:
            self.overlay.set_show_source(on)

    # -- incoming --------------------------------------------------------------
    def set_connected(self, ok: bool) -> None:
        self.conn_label.setText("Đã kết nối" if ok else "Đang kết nối…")
        if self.overlay is not None:
            self.overlay.set_connecting(not ok)

    def set_apps(self, apps: list) -> None:
        self.table.setRowCount(len(apps))
        for row, a in enumerate(apps):
            name = QTableWidgetItem(str(a.get("display_name") or a.get("exe", "")))
            name.setToolTip(str(a.get("exe", "")))
            state = QTableWidgetItem("Đang phát" if a.get("active") else "Im lặng")
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, state)
            box = QCheckBox()
            box.setChecked(bool(a.get("allowed")))
            exe = str(a.get("exe", ""))
            box.toggled.connect(lambda v, e=exe: self._send("set_allowed", exe=e, allowed=v))
            holder = QWidget()
            lay = QHBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(box)
            self.table.setCellWidget(row, 2, holder)

    def apply_config(self, cfg: dict) -> None:
        self._updating = True
        try:
            self._select(self.src_combo, cfg.get("langs", {}).get("src"))
            model = cfg.get("asr", {}).get("model")
            if model and self.model_combo.findData(model) < 0:  # hand-edited model: show, don't override
                self.model_combo.addItem(f"{model} (tùy chỉnh)", model)
            self._select(self.model_combo, model)
            tts = cfg.get("tts", {})
            self.tts_check.setChecked(bool(tts.get("enabled", False)))
            if "volume" in tts and not self.volume.isSliderDown():
                self.volume.setValue(round(float(tts["volume"]) * 100))
            ov = cfg.get("overlay", {})
            if self.overlay is not None and "hold_seconds" in ov:
                self.overlay.set_hold_seconds(float(ov["hold_seconds"]))
        finally:
            self._updating = False

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    def show_status(self, msg: dict) -> None:
        text, style = describe_status(msg)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {_COLORS[style]};")
