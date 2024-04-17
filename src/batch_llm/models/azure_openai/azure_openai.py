import asyncio
import logging
import os
from typing import Any

import openai
from openai import AsyncAzureOpenAI, AzureOpenAI

from batch_llm.base import AsyncBaseModel, BaseModel
from batch_llm.models.azure_openai.azure_openai_utils import process_response
from batch_llm.settings import Settings
from batch_llm.utils import (
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)


class AzureOpenAIModel(BaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        api_version: str = "2023-09-15-preview",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        # try to get the api key and endpoint from the environment variables
        self.api_key = os.environ.get("OPENAI_AZURE_API_KEY")
        self.azure_endpoint = os.environ.get("OPENAI_AZURE_API_ENDPOINT")

        # raise error if the api key or endpoint is not found
        if self.api_key is None:
            raise ValueError("OPENAI_AZURE_API_KEY environment variable not found")
        if self.azure_endpoint is None:
            raise ValueError("OPENAI_AZURE_API_ENDPOINT environment variable not found")

        self.api_type = "azure"
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        openai.api_key = self.api_key
        openai.azure_endpoint = self.azure_endpoint
        openai.api_type = self.api_type
        openai.api_version = self.api_version

    def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

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

        return prompt, generation_config, mode

    def _query_string(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, generation_config, mode = self._obtain_model_inputs(
            prompt_dict
        )

        try:
            if mode == "chat":
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **generation_config,
                )
            elif mode == "query":
                response = self.client.completions.create(
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

    def _query_chat(self, prompt_dict: dict, index: int | str) -> dict:
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
                response = self.client.chat.completions.create(
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

    def query(self, prompt_dict: dict, index: int | str = "NA") -> dict:
        if isinstance(prompt_dict["prompt"], str):
            response_dict = self._query_string(
                prompt_dict=prompt_dict,
                index=index,
            )
        elif isinstance(prompt_dict["prompt"], list):
            response_dict = self._query_chat(
                prompt_dict=prompt_dict,
                index=index,
            )
        else:
            raise TypeError(
                f"If model == 'azure-openai', then prompt must be a string or a list, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict


class AsyncAzureOpenAIModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        api_version: str = "2023-09-15-preview",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        # try to get the api key and endpoint from the environment variables
        self.api_key = os.environ.get("OPENAI_AZURE_API_KEY")
        self.azure_endpoint = os.environ.get("OPENAI_AZURE_API_ENDPOINT")

        # raise error if the api key or endpoint is not found
        if self.api_key is None:
            raise ValueError("OPENAI_AZURE_API_KEY environment variable not found")
        if self.azure_endpoint is None:
            raise ValueError("OPENAI_AZURE_API_ENDPOINT environment variable not found")

        self.api_type = "azure"
        self.api_version = api_version
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        openai.api_key = self.api_key
        openai.azure_endpoint = self.azure_endpoint
        openai.api_type = self.api_type
        openai.api_version = self.api_version

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

    async def async_query(self, prompt_dict: dict, index: int | str = "NA") -> dict:
        if isinstance(prompt_dict["prompt"], str):
            response_dict = await self._async_query_string(
                prompt_dict=prompt_dict,
                index=index,
            )
        elif isinstance(prompt_dict["prompt"], list):
            response_dict = await self._async_query_chat(
                prompt_dict=prompt_dict,
                index=index,
            )
        else:
            raise TypeError(
                f"If model == 'azure-openai', then prompt must be a string or a list, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict
