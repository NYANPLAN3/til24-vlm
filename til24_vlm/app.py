"""Main app."""

from dotenv import load_dotenv

load_dotenv()

import base64
import logging
import os
import sys

from fastapi import FastAPI, Request

from .log import setup_logging
from .VLMManager import VLMManager

__all__ = ["create_app"]

setup_logging()
log = logging.getLogger(__name__)


def create_app():
    """App factory."""
    app = FastAPI()
    vlm_manager = VLMManager()

    @app.get("/hello")
    async def hello():
        """J-H: I added this to dump useful info for debugging.

        Returns:
            dict: JSON message.
        """
        debug = {}
        debug["py_version"] = sys.version
        debug["task"] = "NLP"
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

    @app.post("/identify")
    async def identify(instance: Request):
        """Performs Object Detection and Identification given an image frame and a text query."""
        # get base64 encoded string of image, convert back into bytes
        input_json = await instance.json()

        predictions = []
        for instance in input_json["instances"]:
            # each is a dict with one key "b64" and the value as a b64 encoded string
            image_bytes = base64.b64decode(instance["b64"])

            bbox = vlm_manager.identify(image_bytes, instance["caption"])
            predictions.append(bbox)

        return {"predictions": predictions}

    return app
