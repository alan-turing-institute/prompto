import os
from typing import Any

from ollama import AsyncClient, Client, ResponseError

from batch_llm.models.base import AsyncBaseModel
from batch_llm.models.ollama.ollama_utils import process_response
from batch_llm.settings import Settings
from batch_llm.utils import (
    log_error_response_query,
    log_success_response_query,
    write_log_message,
)


class AsyncOllamaModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        self.ollama_endpoint = os.environ.get("OLLAMA_API_ENDPOINT")

        if self.ollama_endpoint is None:
            raise ValueError("OLLAMA_API_ENDPOINT environment variable not found")

        self.client = Client(host=self.ollama_endpoint)
        self.async_client = AsyncClient(host=self.ollama_endpoint)

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        issues = []

        # check the required environment variables are set
        required_env_vars = ["OLLAMA_API_ENDPOINT"]
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(ValueError(f"Environment variable {var} is not set"))

        # check if the API endpoint is a valid endpoint
        if "OLLAMA_API_ENDPOINT" in os.environ:
            client = Client(host=os.environ["OLLAMA_API_ENDPOINT"])
            try:
                # try to just get the list of models to check if the endpoint is valid
                client.list()
            except Exception as err:
                issues.append(
                    ValueError(
                        f"OLLAMA_API_ENDPOINT is not a valid endpoint: {type(err).__name__} - {err}"
                    )
                )

        # check the optional environment variables are set and warn if not
        other_env_vars = ["OLLAMA_MODEL_NAME"]
        for var in other_env_vars:
            if var not in os.environ:
                issues.append(Warning(f"Environment variable {var} is not set"))

        # check the default model name is a valid model and is downloaded
        if "OLLAMA_MODEL_NAME" in os.environ:
            client = Client(host=os.environ["OLLAMA_API_ENDPOINT"])
            try:
                client.show(model=os.environ["OLLAMA_MODEL_NAME"])
            except Exception as err:
                issues.append(
                    ValueError(
                        f"OLLAMA_MODEL_NAME is not a valid model: {type(err).__name__} - {err}"
                    )
                )

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        issues = []

        # check if the model_name (if provided) is a valid model
        if "model_name" in prompt_dict:
            client = Client(host=os.environ["OLLAMA_API_ENDPOINT"])
            try:
                client.show(model=prompt_dict["model_name"])
            except Exception as err:
                issues.append(
                    ValueError(
                        f"model_name '{prompt_dict['model_name']}' is not a valid model: {type(err).__name__} - {err}"
                    )
                )

        return issues

    def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        model_name = prompt_dict.get("model_name", None) or os.environ.get(
            "OLLAMA_MODEL_NAME"
        )
        if model_name is None:
            log_message = (
                "model_name is not set. Please set the OLLAMA_MODEL_NAME environment variable "
                "or pass the model_name in the prompt dictionary"
            )
            write_log_message(log_file=self.log_file, log_message=log_message, log=True)
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
            response = await self.async_client.generate(
                model=model_name,
                prompt=prompt,
                options=options,
            )

            response_text = process_response(response)

            log_success_response_query(
                index=index,
                model=f"Ollama ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except ResponseError as err:
            if "try pulling it first" in str(err):
                # if there's a response error due to a model not being downloaded,
                # raise a NotImplementedError so that it doesn't get retried
                raise NotImplementedError(
                    f"Model {model_name} is not downloaded: {type(err).__name__} - {err}"
                )
            elif "invalid options" in str(err):
                # if there's a response error due to invalid options, raise a ValueError
                # so that it doesn't get retried
                raise ValueError(
                    f"Invalid options for model {model_name}: {type(err).__name__} - {err}"
                )
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"Ollama ({model_name})",
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
                f"if api == 'ollama', then prompt must be a string, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict
