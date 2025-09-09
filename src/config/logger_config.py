import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --------------------------
# Always place logs/ at project root (next to src/)
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go two levels up from src/config/
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Optional: reduce Azure SDK HTTP spam
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def _console_supports_utf8() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    return enc.lower().replace("-", "") == "utf8"


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with:
      - Rotating file handler (UTF-8) -> logs/<log_file>
      - Console handler (safe for Windows consoles)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging

    if logger.handlers:
        return logger  # handler(s) already attached

    # ---- File handler (UTF-8, rotating) ----
    file_path = LOG_DIR / log_file
    fh = RotatingFileHandler(
        filename=file_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # ---- Console handler ----
    ch = logging.StreamHandler(stream=sys.stdout)

    class StripNonEncodableFilter(logging.Filter):
        """Avoid UnicodeEncodeError in Windows CP1252 consoles."""
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            try:
                msg.encode(enc)
            except UnicodeEncodeError:
                record.msg = msg.encode(enc, errors="replace").decode(enc)
                record.args = ()
            return True

    if not _console_supports_utf8():
        ch.addFilter(StripNonEncodableFilter())

    console_fmt = logging.Formatter("%(levelname)s | %(message)s")
    ch.setFormatter(console_fmt)
    ch.setLevel(level)
    logger.addHandler(ch)

    logging.captureWarnings(True)  # capture warnings too

    return logger


def install_global_exception_hook(logger: logging.Logger) -> None:
    """Log any uncaught exceptions so they don’t vanish."""
    def _handle(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = _handle
