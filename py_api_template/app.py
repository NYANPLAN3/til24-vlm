"""Main app."""

import asyncio
import logging

from dotenv import load_dotenv
from fastapi import FastAPI

__all__ = ["app"]

load_dotenv()

log = logging.getLogger(__name__)

app = FastAPI()


@app.get("/hello")
async def hello():
    """Returns a greeting.

    Returns:
        dict: A greeting message.
    """
    log.warn("zzz... 1 more second...")
    await asyncio.sleep(1)
    log.info("...zzz... oh wha...?!")
    return {"message": "Hello, World!"}
