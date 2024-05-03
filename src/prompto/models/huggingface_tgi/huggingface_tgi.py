import logging
import os
from typing import Any

import openai
from openai import AsyncOpenAI

from prompto.models.base import AsyncBaseModel
from prompto.models.openai.openai import process_response
from prompto.settings import Settings
from prompto.utils import (
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)

API_ENDPOINT_VAR_NAME = "HUGGINGFACE_TGI_API_ENDPOINT"
API_KEY_VAR_NAME = "HUGGINGFACE_TGI_API_KEY"


class AsyncHuggingfaceTGIModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        self.api_type = "tgi"

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        issues = []

        # check the optional environment variables are set and warn if not
        issues.extend(
            check_optional_env_variables_set([API_ENDPOINT_VAR_NAME, API_KEY_VAR_NAME])
        )

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        # for Huggingface TGI, there's specific environment variables that need to be set
        # for different model_name values
        issues = []

        if "model_name" not in prompt_dict:
            # use the default environment variables
            # check the required environment variables are set
            issues.extend(check_required_env_variables_set([API_ENDPOINT_VAR_NAME]))

            # check the optional environment variables are set and warn if not
            issues.extend(check_optional_env_variables_set([API_KEY_VAR_NAME]))
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

            # check the optional environment variables are set and warn if not
            issues.extend(
                check_optional_env_variables_set(
                    [f"{API_KEY_VAR_NAME}_{model_name}", API_KEY_VAR_NAME]
                )
            )

        return issues

    def _obtain_model_inputs(
        self, prompt_dict: dict
    ) -> tuple[str, str, AsyncOpenAI, dict, str]:
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict.get("model_name", None)
        if model_name is None:
            # use the default environment variables
            api_key_env_var = API_KEY_VAR_NAME
            api_endpoint_env_var = API_ENDPOINT_VAR_NAME
        else:
            # use the model specific environment variables if they exist
            api_key_env_var = f"{API_KEY_VAR_NAME}_{model_name}"
            if api_key_env_var not in os.environ:
                api_key_env_var = API_KEY_VAR_NAME

            api_endpoint_env_var = f"{API_ENDPOINT_VAR_NAME}_{model_name}"
            if api_endpoint_env_var not in os.environ:
                api_endpoint_env_var = API_ENDPOINT_VAR_NAME

        API_KEY = os.environ.get(api_key_env_var)
        API_ENDPOINT = os.environ.get(api_endpoint_env_var)

        if API_KEY is None:
            # need pass string to initialise OpenAI client
            API_KEY = "-"

        if API_ENDPOINT is None:
            raise ValueError(f"{api_endpoint_env_var} environment variable not found")

        openai.api_key = API_KEY
        openai.api_type = API_ENDPOINT

        client = AsyncOpenAI(
            base_url=f"{API_ENDPOINT}/v1/",
            api_key=API_KEY,
        )

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
        mode = prompt_dict.get("mode", "completion")
        if mode not in ["chat", "completion"]:
            raise ValueError(f"mode must be 'chat' or 'completion', not {mode}")

        return prompt, model_name, client, generation_config, mode

    async def _async_query_string(self, prompt_dict: dict, index: int | str) -> dict:
        prompt, model_name, client, generation_config, mode = self._obtain_model_inputs(
            prompt_dict
        )

        try:
            if mode == "chat":
                response = await client.chat.completions.create(
                    model=self.api_type,
                    messages=[{"role": "user", "content": prompt}],
                    **generation_config,
                )
            elif mode == "completion":
                response = await client.completions.create(
                    model=self.api_type,
                    prompt=prompt,
                    **generation_config,
                )

            response_text = process_response(response)

            # obtain model name
            prompt_dict["model"] = response.model

            log_success_response_query(
                index=index,
                model=f"Huggingface TGI ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"Huggingface TGI ({model_name})",
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
        prompt, model_name, client, generation_config, _ = self._obtain_model_inputs(
            prompt_dict
        )

        messages = []
        response_list = []
        try:
            for message_index, message in enumerate(prompt):
                # add the user message to the list of messages
                messages.append({"role": "user", "content": message})
                # obtain the response from the model
                response = await client.chat.completions.create(
                    model=self.api_type,
                    messages=messages,
                    **generation_config,
                )
                # parse the response to obtain the response text
                response_text = process_response(response)
                # add the response to the list of responses
                response_list.append(response_text)
                # add the response message to the list of messages
                messages.append({"role": "assistant", "content": response_text})

                # obtain model name
                prompt_dict["model"] = response.model

                log_success_response_chat(
                    index=index,
                    model=f"Huggingface TGI ({model_name})",
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
                model=f"Huggingface TGI ({model_name})",
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
                f"if api == 'huggingface-tgi', then prompt must be a string or a list, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict