from abc import ABC
from typing import Any

from batch_llm.settings import Settings


class BaseModel(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        self.settings = settings
        self.log_file = log_file

    def check_environment_variables(self) -> list[Exception]:
        # method for checking the environment variables
        # returns a list of exceptions or warnings if the environment variables are not set
        raise NotImplementedError

    def check_prompt_dict(self, prompt_dict: dict) -> list[Exception]:
        # method for checking the prompt dictionary
        # returns a list of exceptions or warnings if the prompt dictionary is not valid
        raise NotImplementedError

    def query(
        self, prompt_dict: dict, index: int | str = "NA", *args: Any, **kwargs: Any
    ) -> dict:
        # method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError

    # TODO (maybe): batch_query (for multiple prompts)


class AsyncBaseModel(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        self.settings = settings
        self.log_file = log_file

    def check_environment_variables(self) -> list[Exception]:
        # method for checking the environment variables
        # returns a list of exceptions or warnings if the environment variables are not set
        raise NotImplementedError

    def check_prompt_dict(self, prompt_dict: dict) -> list[Exception]:
        # method for checking the prompt dictionary
        # returns a list of exceptions or warnings if the prompt dictionary is not valid
        raise NotImplementedError

    async def async_query(
        self, prompt_dict: dict, index: int | str = "NA", *args: Any, **kwargs: Any
    ) -> dict:
        # async method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError

    # TODO (maybe): async_batch_query (for multiple prompts asynchronously)
