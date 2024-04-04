from abc import ABC
from typing import Any

from batch_llm.settings import Settings


class BaseModel(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        self.settings = settings
        self.log_file = log_file

    def query(self, prompt: str, *args: Any, **kwargs: Any) -> dict:
        # method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError

    async def async_query(self, prompt: str, *args: Any, **kwargs: Any) -> dict:
        # async method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError

    # TODO (maybe): batch_query (for multiple prompts)

    # TODO (maybe): async_batch_query (for multiple prompts asynchronously)
