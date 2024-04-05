import asyncio
import logging
import random
import time
from typing import Any

from batch_llm.base import BaseModel
from batch_llm.settings import Settings


class TestModel(BaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)

    def query(index, prompt):
        # return an error 1/5 times
        raise_error = random.randint(1, 5) == 1
        if raise_error:
            raise ValueError("This is a test error which we should handle and return")
        else:
            # wait 100 seconds to simulate a long-running task
            time.sleep(100)

        logging.info(
            f"Response recieved (i={index}) \nPrompt: {prompt[:50]}... \nResponse: This is a test response"
        )
        return "This is a test response"

    async def async_query(index, prompt):
        # return an error 1/5 times
        raise_error = random.randint(1, 5) == 1
        if raise_error:
            raise ValueError("This is a test error which we should handle and return")
        else:
            # wait 100 seconds to simulate a long-running task
            await asyncio.sleep(100)

        logging.info(
            f"Response recieved (i={index}) \nPrompt: {prompt[:50]}... \nResponse: This is a test response"
        )
        return "This is a test response"
