from abc import ABC
from typing import Any


class BaseModel(ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

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
