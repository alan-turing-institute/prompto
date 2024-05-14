import os
from typing import Any

import requests

from prompto.models.base import AsyncBaseModel
from prompto.models.quart.quart_utils import (
    async_client_generate,
    get_model_name_identifier,
)
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    log_error_response_query,
    log_success_response_query,
    write_log_message,
)

API_ENDPOINT_VAR_NAME = "QUART_API_ENDPOINT"
MODEL_NAME_VAR_NAME = "QUART_MODEL_NAME"


class AsyncQuartModel(AsyncBaseModel):
    """
    Class for querying the Quart API asynchronously.

    Parameters
    ----------
    settings : Settings
        The settings for the pipeline/experiment
    log_file : str
        The path to the log file
    """

    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        """
        For Quart, there are some optional environment variables:
        - QUART_API_ENDPOINT
        - QUART_MODEL_NAME

        These are optional only if the model_name is passed
        in the prompt dictionary. If the model_name is not
        passed, then the default values are taken from these
        environment variables.

        These are checked in the check_prompt_dict method to ensure that
        the required environment variables are set.

        If QUART_API_ENDPOINT is set, we check if the API endpoint.

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the environment variables
            are not set
        """
        issues = []

        # check the optional environment variables are set and warn if not
        issues.extend(
            check_optional_env_variables_set(
                [API_ENDPOINT_VAR_NAME, MODEL_NAME_VAR_NAME]
            )
        )

        # check if the API endpoint is a valid endpoint
        if API_ENDPOINT_VAR_NAME in os.environ:
            response = requests.get(os.environ[API_ENDPOINT_VAR_NAME])
            if response.status_code != 200:
                issues.append(
                    ValueError(
                        f"{API_ENDPOINT_VAR_NAME} is not working. Status code: {response.status_code}"
                    )
                )
        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        """
        For Quart, we make the following model-specific checks:
        - "prompt" must be a string
        - if "model_name" is not passed in the prompt dictionary,
          then the default environment variables (QUART_API_ENDPOINT,
          QUART_MODEL_NAME) must be set
        - if "model_name" is passed in the prompt dictionary, then
          then for the API endpoint, either the model-specific endpoint
          (QUART_API_ENDPOINT_{identifier}) (where identifier is the
          model name with invalid characters replaced by underscores
          obtained using get_model_name_identifier method) or the
          default endpoint must be set

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to check

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the prompt dictionary
            is not valid
        """
        issues = []

        # check prompt is of the right type
        match prompt_dict["prompt"]:
            case str(_):
                pass
            case _:
                issues.append(
                    TypeError(
                        "if api == 'quart', then prompt must be a string, "
                        f"not {type(prompt_dict['prompt'])}"
                    )
                )

        if "model_name" not in prompt_dict:
            # use the default environment variables
            # check the required environment variables are set
            issues.extend(check_required_env_variables_set([API_ENDPOINT_VAR_NAME]))

        else:
            # use the model specific environment variables
            model_name = prompt_dict["model_name"]
            # replace any invalid characters in the model name
            identifier = get_model_name_identifier(model_name)

            # check the required environment variables are set
            # must either have the model specific endpoint or the default endpoint set
            issues.extend(
                check_either_required_env_variables_set(
                    [[f"{API_ENDPOINT_VAR_NAME}_{identifier}", API_ENDPOINT_VAR_NAME]]
                )
            )

        return issues

    async def _obtain_model_inputs(
        self, prompt_dict: dict
    ) -> tuple[str, str, str, dict]:
        """
        Async method to obtain the model inputs from the prompt dictionary.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model

        Returns
        -------
        tuple[str, str, str, dict]
            A tuple containing the prompt, model name, API endpoint,
            and generation config to use for querying the model
        """
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict.get("model_name", None)
        if model_name is None:
            # use the default environment variables
            QUART_ENDPOINT = API_ENDPOINT_VAR_NAME
        else:
            # use the model specific environment variables if they exist
            # replace any invalid characters in the model name
            identifier = get_model_name_identifier(model_name)
            QUART_ENDPOINT = f"{API_ENDPOINT_VAR_NAME}_{identifier}"
            if QUART_ENDPOINT not in os.environ:
                QUART_ENDPOINT = API_ENDPOINT_VAR_NAME

        quart_endpoint = os.environ.get(QUART_ENDPOINT)

        if quart_endpoint is None:
            raise ValueError(f"{QUART_ENDPOINT} environment variable not found")

        # get parameters dict (if any)
        generation_config = prompt_dict.get("parameters", None)
        if generation_config is None:
            generation_config = {}
        if type(generation_config) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config)}"
            )

        return prompt, model_name, quart_endpoint, generation_config

    async def _async_query_string(self, prompt_dict: dict, index: int | str) -> dict:
        """
        Async method for querying the model with a string prompt
        (prompt_dict["prompt"] is a string),
        i.e. single-turn completion or chat.
        """
        prompt, model_name, quart_endpoint, generation_config = (
            await self._obtain_model_inputs(prompt_dict)
        )

        try:
            response = await async_client_generate(
                data={
                    "text": prompt,
                    "model": model_name,
                    "options": generation_config,
                },
                url=quart_endpoint,
                headers={"Content-Type": "application/json"},
            )

            response_text = response["response"][0]["generated_text"]

            log_success_response_query(
                index=index,
                model=f"Quart ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict

        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"Quart ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file,
                    log_message=log_message,
                    log=True,
                )
            raise err

    async def async_query(self, prompt_dict: dict, index: int | str = "NA") -> dict:
        """
        Async Method for querying the API/model asynchronously.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model
        index : int | str
            The index of the prompt in the experiment

        Returns
        -------
        dict
            Completed prompt_dict with "response" key storing the response(s)
            from the LLM

        Raises
        ------
        Exception
            If an error occurs during the querying process
        """
        match prompt_dict["prompt"]:
            case str(_):
                return await self._async_query_string(
                    prompt_dict=prompt_dict,
                    index=index,
                )
            case _:
                pass

        raise TypeError(
            f"if api == 'quart', then prompt must be a string, "
            f"not {type(prompt_dict['prompt'])}"
        )
