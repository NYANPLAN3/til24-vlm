"""Main app."""

import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI

__all__ = ["app"]

load_dotenv()

log = logging.getLogger(__name__)

app = FastAPI()


@app.get("/hello")
async def hello():
    """J-H: I added this to dump useful info for debugging.

    Returns:
        dict: JSON message.
    """
    debug = {}
    debug["py_version"] = sys.version
    debug["task"] = "ASR"
    debug["env"] = dict(os.environ)

    try:
        import torch  # type: ignore

        debug["torch_version"] = torch.__version__
        debug["cuda_device"] = str(torch.zeros([10, 10], device="cuda").device)
    except ImportError:
        pass

    return debug


@app.get("/health")
async def health():
    """Competition admin needs this."""
    return {"message": "health ok"}
