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
        issues = []

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

    async def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        # obtain the prompt from the prompt dictionary
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
        if isinstance(prompt_dict["prompt"], str):
            response_dict = await self._async_query_string(
                prompt_dict=prompt_dict,
                index=index,
            )
        else:
            raise TypeError(
                f"If model == 'quart', then prompt must be a string, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict
