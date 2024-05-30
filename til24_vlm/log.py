"""Setup logging."""

import logging


def setup_logging():
    """Setup logging configuration if none is present."""
    logging.basicConfig(
        format="%(levelname)s|%(name)s|%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    logging.getLogger("src").setLevel(logging.DEBUG)
