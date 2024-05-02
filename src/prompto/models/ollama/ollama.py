import os
from typing import Any

from ollama import AsyncClient, Client, ResponseError

from prompto.models.base import AsyncBaseModel
from prompto.models.ollama.ollama_utils import process_response
from prompto.settings import Settings
from prompto.utils import (
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    log_error_response_query,
    log_success_response_query,
    write_log_message,
)

API_ENDPOINT_VAR_NAME = "OLLAMA_API_ENDPOINT"
MODEL_NAME_VAR_NAME = "OLLAMA_MODEL_NAME"


class AsyncOllamaModel(AsyncBaseModel):
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
            client = Client(host=os.environ[API_ENDPOINT_VAR_NAME])
            try:
                # try to just get the list of models to check if the endpoint is valid
                client.list()
            except Exception as err:
                issues.append(
                    ValueError(
                        f"{API_ENDPOINT_VAR_NAME} is not a valid endpoint: {type(err).__name__} - {err}"
                    )
                )

        # check the default model name is a valid model and is downloaded
        if MODEL_NAME_VAR_NAME in os.environ:
            client = Client(host=os.environ[API_ENDPOINT_VAR_NAME])
            try:
                client.show(model=os.environ[MODEL_NAME_VAR_NAME])
            except Exception as err:
                issues.append(
                    ValueError(
                        f"{MODEL_NAME_VAR_NAME} is not a valid model: {type(err).__name__} - {err}"
                    )
                )

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        issues = []

        if "model_name" not in prompt_dict:
            # use the default environment variables
            # check the required environment variables are set
            issues.extend(
                check_required_env_variables_set(
                    [API_ENDPOINT_VAR_NAME, MODEL_NAME_VAR_NAME]
                )
            )
        else:
            # use the model specific environment variables
            model_name = prompt_dict["model_name"]

            # check the required environment variables are set
            # must either have the model specific endpoint or the default endpoint set
            issues.extend(
                check_either_required_env_variables_set(
                    [[f"{API_ENDPOINT_VAR_NAME}_{model_name}", API_ENDPOINT_VAR_NAME]]
                )
            )

        return issues

    def _obtain_model_inputs(
        self, prompt_dict: dict
    ) -> tuple[str, str, AsyncClient, dict]:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict.get("model_name", None)
        if model_name is None:
            # use the default environment variables
            model_name = os.environ.get(MODEL_NAME_VAR_NAME)
            if model_name is None:
                log_message = (
                    f"model_name is not set. Please set the {MODEL_NAME_VAR_NAME} "
                    "environment variable or pass the model_name in the prompt dictionary"
                )
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
                raise ValueError(log_message)

            api_endpoint_env_var = API_ENDPOINT_VAR_NAME
        else:
            # use the model specific environment variables if they exist
            api_endpoint_env_var = f"{API_ENDPOINT_VAR_NAME}_{model_name}"
            if api_endpoint_env_var not in os.environ:
                api_endpoint_env_var = API_ENDPOINT_VAR_NAME

        API_ENDPOINT = os.environ.get(api_endpoint_env_var)

        # raise error if the api endpoint is not found
        if API_ENDPOINT is None:
            raise ValueError(f"{api_endpoint_env_var} environment variable not found")

        client = AsyncClient(host=API_ENDPOINT)

        # get parameters dict (if any)
        generation_config = prompt_dict.get("parameters", None)
        if generation_config is None:
            generation_config = {}
        if type(generation_config) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config)}"
            )

        return prompt, model_name, client, generation_config

    async def _async_query_string(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, client, generation_config = self._obtain_model_inputs(
            prompt_dict
        )

        try:
            response = await client.generate(
                model=model_name,
                prompt=prompt,
                options=generation_config,
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
