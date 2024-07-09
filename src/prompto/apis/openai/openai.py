import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from prompto.apis.base import AsyncAPI
from prompto.apis.openai.openai_utils import openai_chat_roles, process_response
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    get_environment_variable,
    get_model_name_identifier,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)

# set names of environment variables
API_KEY_VAR_NAME = "OPENAI_API_KEY"


class OpenAIAPI(AsyncAPI):
    """
    Class for querying the OpenAI API asynchronously.

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
        self.api_type = "openai"

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        """
        For OpenAI, there are some optional environment:
        - OPENAI_API_KEY

        These are optional only if the model_name is passed
        in the prompt dictionary. If the model_name is not
        passed, then the default values are taken from these
        environment variables.

        These are checked in the check_prompt_dict method to ensure that
        the required environment variables are set.

        Returns
        -------
        list[Exception]
            A list of exceptions or warnings if the environment variables
            are not set
        """
        issues = []

        # check the optional environment variables are set and warn if not
        issues.extend(check_optional_env_variables_set([API_KEY_VAR_NAME]))

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        """
        For OpenAI, we make the following model-specific checks:
        - "prompt" key must be of type str, list[str], or list[dict[str,str]]
        - model-specific environment variable (OPENAI_API_KEY_{identifier})
          (where identifier is the model name with invalid characters replaced
          by underscores obtained using get_model_name_identifier function)
          can be set or the default environment variable must be set
        - if "mode" is passed, it must be one of 'chat' or 'completion'

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
        if isinstance(prompt_dict["prompt"], str):
            pass
        elif isinstance(prompt_dict["prompt"], list):
            if all([isinstance(message, str) for message in prompt_dict["prompt"]]):
                pass
            if all(
                isinstance(message, dict) for message in prompt_dict["prompt"]
            ) and all(
                [
                    set(d.keys()) == {"role", "content"}
                    and d["role"] in openai_chat_roles
                    for d in prompt_dict["prompt"]
                ]
            ):
                pass
        else:
            issues.append(
                TypeError(
                    "if api == 'openai', then the prompt must be a str, list[str], or "
                    "list[dict[str,str]] where the dictionary contains the keys 'role' and "
                    "'content' only, and the values for 'role' must be one of 'system', 'user' or "
                    "'assistant'"
                )
            )

        # use the model specific environment variables if they exist
        model_name = prompt_dict["model_name"]
        # replace any invalid characters in the model name
        identifier = get_model_name_identifier(model_name)

        # check the required environment variables are set
        # must either have the model specific key or the default key set
        issues.extend(
            check_either_required_env_variables_set(
                [
                    [f"{API_KEY_VAR_NAME}_{identifier}", API_KEY_VAR_NAME],
                ]
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

    async def _obtain_model_inputs(
        self, prompt_dict: dict
    ) -> tuple[str, str, AsyncOpenAI, dict, str]:
        """
        Async method to obtain the model inputs from the prompt dictionary.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model

        Returns
        -------
        tuple[str, str, AsyncAzureOpenAI, dict, str]
            A tuple containing the prompt, model name, AzureOpenAI client object,
            the generation config, and mode to use for querying the model
        """
        # obtain the prompt from the prompt dictionary
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict["model_name"]
        api_key = get_environment_variable(
            env_variable=API_KEY_VAR_NAME, model_name=model_name
        )

        openai.api_key = api_key
        openai.api_type = self.api_type
        client = AsyncOpenAI(api_key=api_key, max_retries=1)

        # get parameters dict (if any)
        generation_config = prompt_dict.get("parameters", None)
        if generation_config is None:
            generation_config = {}
        if type(generation_config) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config)}"
            )

        # obtain mode (default is chat)
        mode = prompt_dict.get("mode", "chat")
        if mode not in ["chat", "completion"]:
            raise ValueError(f"mode must be one of 'chat' or 'completion', not {mode}")

        return prompt, model_name, client, generation_config, mode

    async def _query_string(self, prompt_dict: dict, index: int | str) -> dict:
        """
        Async method for querying the model with a string prompt
        (prompt_dict["prompt"] is a string),
        i.e. single-turn completion or chat.
        """
        prompt, model_name, client, generation_config, mode = (
            await self._obtain_model_inputs(prompt_dict)
        )

        try:
            if mode == "chat":
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **generation_config,
                )
            elif mode == "completion":
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **generation_config,
                )

            response_text = process_response(response)

            log_success_response_query(
                index=index,
                model=f"OpenAI ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"OpenAI ({model_name})",
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

    async def _query_chat(self, prompt_dict: dict, index: int | str) -> dict:
        """
        Async method for querying the model with a chat prompt
        (prompt_dict["prompt"] is a list of strings to sequentially send to the model),
        i.e. multi-turn chat with history.
        """
        prompt, model_name, client, generation_config, _ = (
            await self._obtain_model_inputs(prompt_dict)
        )

        messages = []
        response_list = []
        try:
            for message_index, message in enumerate(prompt):
                # add the user message to the list of messages
                messages.append({"role": "user", "content": message})
                # obtain the response from the model
                response = await client.chat.completions.create(
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
                    model=f"OpenAI ({model_name})",
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
                model=f"OpenAI ({model_name})",
                message_index=message_index,
                n_messages=len(prompt),
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file,
                    log_message=log_message,
                    log=True,
                )
            raise err

    async def _query_history(self, prompt_dict: dict, index: int | str) -> dict:
        """
        Async method for querying the model with a chat prompt with history
        (prompt_dict["prompt"] is a list of dictionaries with keys "role" and "content",
        where "role" is one of "user", "assistant", or "system" and "content" is the message),
        i.e. multi-turn chat with history.
        """
        prompt, model_name, client, generation_config, _ = (
            await self._obtain_model_inputs(prompt_dict)
        )

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=prompt,
                **generation_config,
            )

            response_text = process_response(response)

            log_success_response_query(
                index=index,
                model=f"OpenAI ({model_name})",
                prompt=prompt,
                response_text=response_text,
            )

            prompt_dict["response"] = response_text
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"OpenAI ({model_name})",
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

    async def query(self, prompt_dict: dict, index: int | str = "NA") -> dict:
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
        if isinstance(prompt_dict["prompt"], str):
            return await self._query_string(
                prompt_dict=prompt_dict,
                index=index,
            )
        elif isinstance(prompt_dict["prompt"], list):
            if all([isinstance(message, str) for message in prompt_dict["prompt"]]):
                return await self._query_chat(
                    prompt_dict=prompt_dict,
                    index=index,
                )
            if all(
                isinstance(message, dict) for message in prompt_dict["prompt"]
            ) and all(
                [
                    set(d.keys()) == {"role", "content"}
                    and d["role"] in openai_chat_roles
                    for d in prompt_dict["prompt"]
                ]
            ):
                return await self._query_history(
                    prompt_dict=prompt_dict,
                    index=index,
                )

        raise TypeError(
            "if api == 'openai', then the prompt must be a str, list[str], or "
            "list[dict[str,str]] where the dictionary contains the keys 'role' and "
            "'content' only, and the values for 'role' must be one of 'system', 'user' or "
            "'assistant'"
        )
