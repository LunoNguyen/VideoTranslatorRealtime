"""Entry point: python -m backend."""
from __future__ import annotations

import logging
import socket
import sys
from pathlib import Path

from backend.config import LOOPBACK, load_config

ROOT = Path(__file__).resolve().parents[1]
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(cfg: dict, root: Path) -> None:
    """Console logging always; logs/backend.log only when logging.enabled (opt-in).

    Log calls must never include transcript/subtitle text.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if cfg.get("logging", {}).get("enabled"):
        (root / "logs").mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(root / "logs" / "backend.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def main(root: Path | None = None) -> None:
    import uvicorn

    from backend.server.app import create_app, default_services

    root = Path(root) if root else ROOT
    cfg = load_config(root)
    setup_logging(cfg, root)
    host, port = LOOPBACK, int(cfg["server"]["port"])  # never anything but loopback
    if not _port_free(host, port):
        print(
            f"Cổng {port} đang được sử dụng (port {port} in use); "
            "hãy đóng chương trình khác hoặc đổi server.port trong config.local.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    app = create_app(cfg, default_services(root))
    logging.getLogger("backend").info("listening on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
