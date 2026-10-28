"""UI entry point: python -m ui [--port N]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from backend.config import load_config
from ui.main_window import MainWindow
from ui.overlay import Overlay
from ui.ws_client import WSClient

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    port = args.port or int(load_config(ROOT)["server"]["port"])

    app = QApplication(sys.argv[:1])
    client = WSClient(f"ws://127.0.0.1:{port}/ws", autoconnect=False)
    overlay = Overlay()
    win = MainWindow(client, overlay)
    client.subtitle.connect(lambda m: overlay.show_subtitle(str(m.get("vi_text", "")), m.get("src_text")))
    overlay.set_connecting(True)
    win.show()
    client.start()
    code = app.exec()
    client.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
