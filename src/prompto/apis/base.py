from abc import ABC
from typing import Any

from prompto.settings import Settings


class AsyncAPI(ABC):
    def __init__(self, settings: Settings, log_file: str, *args: Any, **kwargs: Any):
        """
        Base class for asynchronous API models.

        Each subclass should implement the following methods:
        - check_environment_variables: a static method that checks
          if the required or optional environment variables are set
        - check_prompt_dict: a static method that checks if an input
          dictionary (prompt_dict) is valid
        - query: an async method that queries the API/model and
          returns the response as a completed dictionary (prompt_dict)

        Parameters
        ----------
        settings : Settings
            The settings for the pipeline/experiment
        log_file : str
            The path to the log file
        """
        self.settings = settings
        self.log_file = log_file

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        """
        Method for checking the environment variables.
        Each subclass should implement this method to check if the
        required or optional environment variables are set.

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the environment variables
            are not set

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass
        """
        raise NotImplementedError(
            "'check_environment_variables' method needs to be implemented by a subclass of AsyncAPI"
        )

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        """
        Method for checking the prompt dictionary.
        Each subclass should implement this method to check if the
        prompt dictionary is a valid input for the model.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to check

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the prompt dictionary
            is not valid

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass
        """
        raise NotImplementedError(
            "'check_prompt_dict' method needs to be implemented by a subclass of AsyncAPI"
        )

    async def query(
        self, prompt_dict: dict, index: int | str = "NA", *args: Any, **kwargs: Any
    ) -> dict:
        """
        Method for querying the API/model asynchronously.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model
        index : int | str, optional
            The index of the prompt in the experiment, by default "NA"

        Returns
        -------
        dict
            Completed prompt_dict with "response" key storing the response(s)
            from the LLM

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass
        """
        # async method for querying the model using the prompt and other arguments
        # returns a dictionary/json object which is saved into the output jsonl
        raise NotImplementedError(
            "'query' method needs to be implemented by a subclass of AsyncAPI"
        )

    # TODO (maybe): async_batch_query (for multiple prompts asynchronously)
