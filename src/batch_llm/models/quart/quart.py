import json
import os
from typing import Any

import requests

from batch_llm.models.base import AsyncBaseModel
from batch_llm.models.quart.quart_utils import async_client_generate
from batch_llm.settings import Settings
from batch_llm.utils import (
    log_error_response_query,
    log_success_response_query,
    write_log_message,
)


class AsyncQuartModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        self.quart_endpoint = os.environ.get("QUART_API_ENDPOINT")

        if self.quart_endpoint is None:
            raise ValueError("QUART_API_ENDPOINT environment variable not found")

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        issues = []

        # check the required environment variables are set

        required_env_vars = ["QUART_API_ENDPOINT"]
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(ValueError(f"Environment variable {var} is not set"))

        # check the optional environment variables are set and warn if not
        other_env_vars = ["QUART_MODEL_NAME"]
        for var in other_env_vars:
            if var not in os.environ:
                issues.append(Warning(f"Environment variable {var} is not set"))

        # check if the API endpoint is a valid endpoint
        if "QUART_API_ENDPOINT" in os.environ:
            response = requests.get(os.environ["QUART_API_ENDPOINT"])

            if response.status_code != 200:

                issues.append(
                    ValueError(
                        f"QUART_API_ENDPOINT is not working. Status code:: {response.status_code}"
                    )
                )
        return issues

    def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        model_name = prompt_dict.get("model_name", None) or os.environ.get(
            "QUART_MODEL_NAME"
        )
        if model_name is None:
            log_message = (
                "model_name is not set. Please set the QUART_MODEL_NAME environment variable "
                "or pass the model_name in the prompt dictionary"
            )
            write_log_message(log_file=self.log_file, log_message=log_message, log=True)
            raise ValueError(log_message)

        else:
            headers = {"Content-Type": "application/json"}
            data = {"text": "Test", "model": model_name}

            response = requests.post(
                os.environ["QUART_API_ENDPOINT"], headers=headers, data=json.dumps(data)
            )
            if response.status_code != 200:
                log_message = f"{model_name} is not a valid model."
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
                raise ValueError(log_message)

        # get parameters dict (if any)
        options = prompt_dict.get("parameters", None)
        if options is None:
            options = {}
        if type(options) is not dict:
            raise TypeError(f"parameters must be a dictionary, not {type(options)}")

        return prompt, model_name, options

    async def _async_query_string(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, options = self._obtain_model_inputs(prompt_dict)

        try:
            response = await async_client_generate(
                data={"text": prompt, "model": model_name, "options": options},
                url=self.quart_endpoint,
                headers={"Content-Type": "application/json"},
            )

            response_text = response["response"]

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
