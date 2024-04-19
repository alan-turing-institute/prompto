from abc import ABC
from typing import Any

from batch_llm.settings import Settings


class BaseModel(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        self.settings = settings
        self.log_file = log_file

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        # method for checking the environment variables
        # returns a list of exceptions or warnings if the environment variables are not set
        raise NotImplementedError(
            "'check_environment_variables' method needs to be implemented by a subclass of BaseModel"
        )

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        # method for checking the prompt dictionary
        # returns a list of exceptions or warnings if the prompt dictionary is not valid
        raise NotImplementedError(
            "'check_prompt_dict' method needs to be implemented by a subclass of BaseModel"
        )

    def query(
        self, prompt_dict: dict, index: int | str = "NA", *args: Any, **kwargs: Any
    ) -> dict:
        # method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError(
            "'query' method needs to be implemented by a subclass of BaseModel"
        )

    # TODO (maybe): batch_query (for multiple prompts)


class AsyncBaseModel(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        self.settings = settings
        self.log_file = log_file

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        # method for checking the environment variables
        # returns a list of exceptions or warnings if the environment variables are not set
        raise NotImplementedError(
            "'check_environment_variables' method needs to be implemented by a subclass of AsyncBaseModel"
        )

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        # method for checking the prompt dictionary
        # returns a list of exceptions or warnings if the prompt dictionary is not valid
        raise NotImplementedError(
            "'check_prompt_dict' method needs to be implemented by a subclass of AsyncBaseModel"
        )

    async def async_query(
        self, prompt_dict: dict, index: int | str = "NA", *args: Any, **kwargs: Any
    ) -> dict:
        # async method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError(
            "'async_query' method needs to be implemented by a subclass of AsyncBaseModel"
        )

    # TODO (maybe): async_batch_query (for multiple prompts asynchronously)
