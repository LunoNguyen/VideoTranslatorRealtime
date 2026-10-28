"""Launcher: start the backend subprocess, wait for its port, run the UI, then stop the backend."""
from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from backend.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent
WAIT_S = 30.0
TAIL_BYTES = 4096


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def stop_process(proc: subprocess.Popen) -> None:
    """Terminate, wait 5 s, then kill."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def show_error(text: str) -> None:
    """Vietnamese error dialog (falls back to stderr)."""
    print(text, file=sys.stderr)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, "Dịch video thời gian thực", text)
        del app
    except Exception:  # noqa: BLE001
        pass


def port_busy_message(port: int) -> str:
    return f"Cổng {port} đang bị chiếm bởi chương trình khác. Hãy đóng nó hoặc đổi server.port trong config.local.json."


def _tail(f) -> str:
    """Last TAIL_BYTES of the stderr file (never loads the whole file)."""
    f.flush()
    f.seek(0, 2)
    f.seek(max(0, f.tell() - TAIL_BYTES))
    return f.read().decode("utf-8", "replace").strip()[-TAIL_BYTES:]


def main() -> int:
    try:
        port = int(load_config(ROOT)["server"]["port"])
    except ConfigError as e:
        show_error(str(e))
        return 1
    if port_open("127.0.0.1", port):  # never attach to a foreign process
        show_error(port_busy_message(port))
        return 1
    with tempfile.TemporaryFile() as err:  # stderr tail only; backend logs go to its own logging
        proc = subprocess.Popen([sys.executable, "-m", "backend"], cwd=ROOT, stderr=err)
        try:
            deadline = time.monotonic() + WAIT_S
            while not port_open("127.0.0.1", port):
                if proc.poll() is not None:
                    show_error(f"Dịch vụ nền đã dừng (mã {proc.returncode}).\n\n{_tail(err)}")
                    return proc.returncode or 1
                if time.monotonic() > deadline:
                    show_error(f"Dịch vụ nền không mở cổng {port} sau {int(WAIT_S)} giây.\n\n{_tail(err)}")
                    return 1
                time.sleep(0.2)
            from ui.__main__ import main as ui_main

            return ui_main(["--port", str(port)])
        finally:
            stop_process(proc)


if __name__ == "__main__":
    sys.exit(main())
