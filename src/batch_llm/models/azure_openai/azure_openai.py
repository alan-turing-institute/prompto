import logging
import os
from typing import Any

import openai
from openai import AsyncAzureOpenAI

from batch_llm.models.base import AsyncBaseModel
from batch_llm.models.openai.openai import process_response
from batch_llm.models.openai.openai_utils import ChatRoles
from batch_llm.settings import Settings
from batch_llm.utils import (
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)

AZURE_API_VERSION_DEFAULT = "2024-02-01"


class AsyncAzureOpenAIModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        api_version: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        # try to get the api key and endpoint from the environment variables
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT")

        # raise error if the api key or endpoint is not found
        if self.api_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not found")
        if self.azure_endpoint is None:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT environment variable not found")

        self.api_type = "azure"
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        if self.api_version is None:
            self.api_version = AZURE_API_VERSION_DEFAULT

        openai.api_key = self.api_key
        openai.azure_endpoint = self.azure_endpoint
        openai.api_type = self.api_type
        openai.api_version = self.api_version
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        issues = []

        # check the required environment variables are set
        required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_ENDPOINT"]
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(ValueError(f"Environment variable {var} is not set"))

        # check the optional environment variables are set and warn if not
        other_env_vars = ["AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_MODEL_NAME"]
        for var in other_env_vars:
            if var not in os.environ:
                issues.append(Warning(f"Environment variable {var} is not set"))

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        issues = []

        # check prompt is of the right type
        match prompt_dict["prompt"]:
            case str(_):
                pass
            case [str(_)]:
                pass
            case [{"role": role, "content": _}, *rest]:
                if role in ChatRoles and all(
                    [
                        set(d.keys()) == {"role", "content"} and d["role"] in ChatRoles
                        for d in rest
                    ]
                ):
                    pass
            case _:
                issues.append(
                    TypeError(
                        "if api == 'azure-openai', then the prompt must be a str, list[str], or "
                        "list[dict[str,str]] where the dictionary contains the keys 'role' and "
                        "'content' only, and the values for 'role' must be one of 'system', 'user' or "
                        "'assistant'"
                    )
                )

        # check if the model_name is set as an environment variable if not provided in the prompt_dict
        if "model_name" not in prompt_dict:
            req_var = "AZURE_OPENAI_MODEL_NAME"
            if req_var not in os.environ:
                issues.append(
                    ValueError(
                        f"model_name is not set and environment variable {req_var} is not set"
                    )
                )

        # if mode is passed, check it is a valid value
        if "mode" in prompt_dict and prompt_dict["mode"] not in ["chat", "completion"]:
            issues.append(
                ValueError(
                    f"Invalid mode value. Must be 'chat' or 'completion', not {prompt_dict['mode']}"
                )
            )

        # TODO: add checks for prompt_dict["parameters"] being
        # valid arguments for OpenAI API without hardcoding

        return issues

    def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict.get("model_name", None) or os.environ.get(
            "AZURE_OPENAI_MODEL_NAME"
        )
        if model_name is None:
            log_message = (
                "model_name is not set. Please set the AZURE_OPENAI_MODEL_NAME environment variable "
                "or pass the model_name in the prompt dictionary"
            )
            write_log_message(log_file=self.log_file, log_message=log_message, log=True)
            raise ValueError(log_message)

        # get parameters dict (if any)
        generation_config = prompt_dict.get("parameters", None)
        if generation_config is None:
            generation_config = {}
        if type(generation_config) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config)}"
            )

        # add in default parameters
        default_generation_config = {
            "max_tokens": 2048,
            "temperature": 0.7,
            "n": 1,
        }
        for key, value in default_generation_config.items():
            if key not in generation_config:
                generation_config[key] = value

        # obtain mode (default is chat)
        mode = prompt_dict.get("mode", "chat")

        return prompt, model_name, generation_config, mode

    async def _async_query_string(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, generation_config, mode = self._obtain_model_inputs(
            prompt_dict
        )

        try:
            if mode == "chat":
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **generation_config,
                )
            elif mode == "query":
                response = await self.client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **generation_config,
                )

            response_text = process_response(response)

            log_success_response_query(
                index=index,
                model=f"AzureOpenAI ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"AzureOpenAI ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.log_file,
                log_message=log_message,
                log=True,
            )
            raise err

    async def _async_query_chat(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, generation_config, mode = self._obtain_model_inputs(
            prompt_dict
        )

        messages = []
        response_list = []
        try:
            for message_index, message in enumerate(prompt):
                # add the user message to the list of messages
                messages.append({"role": "user", "content": message})
                # obtain the response from the model
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **generation_config,
                )
                # parse the response to obtain the response text
                response_text = process_response(response)
                # add the response to the list of responses
                response_list.append(response_text)
                # add the response message to the list of messages
                messages.append({"role": "assistant", "content": response_text})

                log_success_response_chat(
                    index=index,
                    model=f"AzureOpenAI ({model_name})",
                    message_index=message_index,
                    n_messages=len(prompt),
                    message=message,
                    response_text=response_text,
                )

            logging.info(f"Chat completed (i={index})")

            prompt_dict["response"] = response_list
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_chat(
                index=index,
                model=f"AzureOpenAI ({model_name})",
                message_index=message_index,
                n_messages=len(prompt),
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.log_file,
                log_message=log_message,
                log=True,
            )
            raise err

    async def _async_query_history(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, generation_config, mode = self._obtain_model_inputs(
            prompt_dict
        )

        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=prompt,
                **generation_config,
            )

            response_text = process_response(response)

            log_success_response_query(
                index=index,
                model=f"AzureOpenAI ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"AzureOpenAI ({model_name})",
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
        match prompt_dict["prompt"]:
            case str(_):
                return await self._async_query_string(
                    prompt_dict=prompt_dict,
                    index=index,
                )
            case [str(_)]:
                return await self._async_query_chat(
                    prompt_dict=prompt_dict,
                    index=index,
                )
            case [{"role": role, "content": _}, *rest]:
                if role in ChatRoles and all(
                    [
                        set(d.keys()) == {"role", "content"} and d["role"] in ChatRoles
                        for d in rest
                    ]
                ):
                    return await self._async_query_history(
                        prompt_dict=prompt_dict,
                        index=index,
                    )
            case _:
                pass

        raise TypeError(
            "if api == 'azure-openai', then the prompt must be a str, list[str], or "
            "list[dict[str,str]] where the dictionary contains the keys 'role' and "
            "'content' only, and the values for 'role' must be one of 'system', 'user' or "
            "'assistant'"
        )
