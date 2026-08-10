import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger("vlm")
    if log.handlers:
        return log
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(level)
    return log


def get_logger() -> logging.Logger:
    return setup_logging()
