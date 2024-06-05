"""Setup logging."""

import logging

import colorlog


def setup_logging():
    """Setup logging configuration if none is present."""
    if not logging.getLogger().hasHandlers():
        lvl = "%(log_color)s%(levelname)-4s%(reset)s"
        s = "%(bold_light_blue)s|%(reset)s"
        tm = "%(thin_cyan)s%(asctime)s%(reset)s"
        nm = "%(thin_cyan)s%(name)-20s%(reset)s"
        msg = "%(light_white)s%(message)s"
        colorlog.basicConfig(
            format=f"{lvl}{s}{tm}{s}{nm} {msg}",
            datefmt="%H:%M:%S",
            force=False,
        )
        logging.addLevelName(logging.DEBUG, "DBG")
        logging.addLevelName(logging.INFO, "INFO")
        logging.addLevelName(logging.WARNING, "WARN")
        logging.addLevelName(logging.ERROR, "ERR")
        logging.addLevelName(logging.CRITICAL, "CRIT")
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    logging.getLogger("til24_vlm").setLevel(logging.DEBUG)
