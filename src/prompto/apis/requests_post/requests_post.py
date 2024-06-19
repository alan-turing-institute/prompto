import os
from typing import Any

import requests

from prompto.apis.base import AsyncBaseAPI
from prompto.apis.requests_post.requests_post_utils import async_client_generate
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    get_model_name_identifier,
    log_error_response_query,
    log_success_response_query,
    write_log_message,
)

API_ENDPOINT_VAR_NAME = "REQUESTS_POST_API_ENDPOINT"


class AsyncRequestsPostAPI(AsyncBaseAPI):
    """
    Class for querying the an API endpoint via a
    requests.post call asynchronously.

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
        For the RequestsPost API, there are some optional environment variables:
        - REQUESTS_POST_API_ENDPOINT

        These are optional only if the model_name is passed
        in the prompt dictionary. If the model_name is not
        passed, then the default values are taken from these
        environment variables.

        These are checked in the check_prompt_dict method to ensure that
        the required environment variables are set.

        If REQUESTS_POST_API_ENDPOINT is set, we check if the API endpoint.

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the environment variables
            are not set
        """
        issues = []

        # check the optional environment variables are set and warn if not
        issues.extend(check_optional_env_variables_set([API_ENDPOINT_VAR_NAME]))

        # check if the API endpoint is a valid endpoint
        if API_ENDPOINT_VAR_NAME in os.environ:
            response = requests.post(os.environ[API_ENDPOINT_VAR_NAME])
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
        For the RequestsPost API, we make the following model-specific checks:
        - "prompt" must be a string
        - if "model_name" is not passed in the prompt dictionary,
          then the default environment variables (REQUESTS_POST_API_ENDPOINT)
          must be set
        - if "model_name" is passed in the prompt dictionary, then
          then for the API endpoint, either the model-specific endpoint
          (REQUESTS_POST_API_ENDPOINT_{identifier}) (where identifier is the
          model name with invalid characters replaced by underscores
          obtained using get_model_name_identifier function) or the
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
            case {}:
                pass
            case _:
                issues.append(
                    TypeError(
                        "if api == 'requests-post', then prompt must be a dictionary, "
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
    ) -> tuple[dict, str, str, dict]:
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
            REQUESTS_POST_ENDPOINT = API_ENDPOINT_VAR_NAME
        else:
            # use the model specific environment variables if they exist
            # replace any invalid characters in the model name
            identifier = get_model_name_identifier(model_name)

            REQUESTS_POST_ENDPOINT = f"{API_ENDPOINT_VAR_NAME}_{identifier}"
            if REQUESTS_POST_ENDPOINT not in os.environ:
                REQUESTS_POST_ENDPOINT = API_ENDPOINT_VAR_NAME

        requests_endpoint = os.environ.get(REQUESTS_POST_ENDPOINT)

        if requests_endpoint is None:
            raise ValueError(f"{REQUESTS_POST_ENDPOINT} environment variable not found")

        # get headers dict (if any)
        headers = prompt_dict.get("headers", None)
        if headers is None:
            headers = {}
        if type(headers) is not dict:
            raise TypeError(f"headers must be a dictionary, not {type(headers)}")

        return prompt, model_name, requests_endpoint, headers

    async def _async_query_dict(self, prompt_dict: dict, index: int | str) -> dict:
        """
        Async method for querying the model with a dictionary prompt
        (prompt_dict["prompt"] is a dictionary which is the data sent).
        """
        prompt, model_name, requests_endpoint, headers = (
            await self._obtain_model_inputs(prompt_dict)
        )

        try:
            response = await async_client_generate(
                data=prompt,
                url=requests_endpoint,
                headers=headers,
            )

            log_success_response_query(
                index=index,
                model=f"RequestsPost ({model_name})",
                prompt=prompt,
                response_text=str(response),
            )

            prompt_dict["response"] = str(response)
            return prompt_dict

        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"RequestsPost ({model_name})",
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
            case {}:
                return await self._async_query_dict(
                    prompt_dict=prompt_dict,
                    index=index,
                )
            case _:
                pass

        raise TypeError(
            f"if api == 'requests-post', then prompt must be a dictionary, "
            f"not {type(prompt_dict['prompt'])}"
        )
