import logging
import os
from typing import Any

from google.genai import Client

# import google.generativeai as genai
# from google.generativeai import GenerativeModel
# from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from prompto.apis.base import AsyncAPI
from prompto.apis.gemini.gemini_utils import (
    convert_history_dict_to_content,
    gemini_chat_roles,
    parse_parts,
    process_response,
    process_safety_attributes,
)
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    check_either_required_env_variables_set,
    check_optional_env_variables_set,
    get_environment_variable,
    get_model_name_identifier,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)

API_KEY_VAR_NAME = "GEMINI_API_KEY"

TYPE_ERROR = TypeError(
    "if api == 'gemini', then the prompt must be a str, list[str], or "
    "list[dict[str,str]] where the dictionary contains the keys 'role' and "
    "'parts' only, and the values for 'role' must be one of 'user' or 'model', "
    "except for the first message in the list of dictionaries can be a "
    "system message with the key 'role' set to 'system'."
)

BLOCKED_SAFETY_ATTRIBUTES = {
    "blocked": "True",
    "finish_reason": "block_reason: OTHER",
}


class GeminiAPI(AsyncAPI):
    """
    Class for asynchronous querying of the Gemini API.

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
        self._clients: dict[str, Client] = {}

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        """
        For Gemini, there are some optional variables:
        - GEMINI_API_KEY

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
        For Gemini, we make the following model-specific checks:
        - "prompt" must be a string or a list of strings
        - model-specific environment variables (GEMINI_API_KEY_{identifier})
          (where identifier is the model name with invalid characters replaced by
          underscores obtained using get_model_name_identifier function) can be optionally set.
        - if "safety_filter" is provided, check that it's one of the valid options
          ("none", "few", "some", "default", "most")
        - if "generation_config" is provided, check that it can create a valid
          google.generativeai.types.GenerationConfig object

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
            elif (
                all(isinstance(message, dict) for message in prompt_dict["prompt"])
                and (
                    set(prompt_dict["prompt"][0].keys()) == {"role", "parts"}
                    and prompt_dict["prompt"][0]["role"]
                    in list(gemini_chat_roles) + ["system"]
                )
                and all(
                    [
                        set(d.keys()) == {"role", "parts"}
                        and d["role"] in gemini_chat_roles
                        for d in prompt_dict["prompt"][1:]
                    ]
                )
            ):
                pass
            else:
                issues.append(TYPE_ERROR)
        else:
            issues.append(TYPE_ERROR)

        # use the model specific environment variables
        model_name = prompt_dict["model_name"]
        # replace any invalid characters in the model name
        identifier = get_model_name_identifier(model_name)

        # check the required environment variables are set
        issues.extend(
            check_either_required_env_variables_set(
                [
                    [f"{API_KEY_VAR_NAME}_{identifier}", API_KEY_VAR_NAME],
                ]
            )
        )

        # check the parameter settings are valid
        # if safety_filter is provided, check that it's one of the valid options
        if "safety_filter" in prompt_dict and prompt_dict["safety_filter"] not in [
            "none",
            "few",
            "some",
            "default",
            "most",
        ]:
            issues.append(ValueError("Invalid safety_filter value"))

        # if generation_config is provided, check that it can create a valid GenerationConfig object
        if "parameters" in prompt_dict:
            try:
                GenerationConfig(**prompt_dict["parameters"])
            except Exception as err:
                issues.append(Exception(f"Invalid generation_config parameter: {err}"))

        return issues

    def _get_client(self, model_name) -> Client:
        """
        Method to get the client for the Gemini API. A separate client is created for each model name, to allow for
        model-specific API keys to be used.

        The client is created only once per model name and stored in the clients dictionary.
        If the client is already created, it is returned from the dictionary.

        Parameters
        ----------
        model_name : str
            The name of the model to use

        Returns
        -------
        Client
            A client for the Gemini API
        """
        # If the Client does not exist, create it
        # already_created = True
        # api_key = "NO VALUE HAS BEEN SET YET"
        # print(f"GeminiAPI: {model_name=}")
        # print(f"GeminiAPI: {self._clients=}")
        # print(f"GeminiAPI: {self._clients.get(model_name, "not found")=}")
        if model_name not in self._clients:
            # already_created = False
            api_key = get_environment_variable(
                env_variable=API_KEY_VAR_NAME, model_name=model_name
            )
            # print(f"Creating client for {model_name} with {api_key=}")
            self._clients[model_name] = Client(api_key=api_key)

        # print(f"Client for {model_name} already created: {already_created}")
        # print(f"{api_key=}")
        # for env_var_name, env_var_val in os.environ.items():
        #     print(f"{env_var_name}={env_var_val}")

        # Return the client for the model name
        return self._clients[model_name]

    async def _obtain_model_inputs(
        self, prompt_dict: dict, system_instruction: str | None = None
    ) -> tuple[str, str, Client, GenerateContentConfig, list | None]:
        """
        Async method to obtain the model inputs from the prompt dictionary.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model]
        system_instruction : str | None
            The system instruction to use for querying the model if any,
            defaults to None

        Returns
        -------
        tuple[str, str, Client, GenerateContentConfig, list | None]
            A tuple containing:
            - the prompt,
            - model name,
            - Client instance,
            - GenerateContentConfig instance (which incorporates the safety settings),
            - (optional) list of multimedia parts (if passed) to use for querying the model or None
        """
        prompt = prompt_dict["prompt"]

        # obtain model name
        model_name = prompt_dict["model_name"]
        client = self._get_client(model_name)

        # define safety settings
        safety_filter = prompt_dict.get("safety_filter", None)
        if safety_filter is None:
            safety_filter = "default"

        # explicitly set the safety settings
        if safety_filter == "none":
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        elif safety_filter == "few":
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ]
        elif safety_filter in ["default", "some"]:
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
            ]
        elif safety_filter == "most":
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
            ]
        else:
            raise ValueError(
                f"safety_filter '{safety_filter}' not recognised. Must be one of: "
                f"none', 'few', 'default'/'some', 'most'"
            )

        # get parameters dict (if any)
        generation_config_params = prompt_dict.get("parameters", None)
        if generation_config_params is None:
            generation_config_params = {}
        if type(generation_config_params) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config_params)}"
            )

        gen_content_config = GenerateContentConfig(
            **generation_config_params,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )

        return prompt, model_name, client, gen_content_config, None

    async def _query_string(self, prompt_dict: dict, index: int | str):
        """
        Async method for querying the model with a string prompt
        (prompt_dict["prompt"] is a string),
        i.e. single-turn completion or chat.
        """
        prompt, model_name, client, generation_config, _ = (
            await self._obtain_model_inputs(
                prompt_dict=prompt_dict, system_instruction=None
            )
        )

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            response_text = process_response(response)
            safety_attributes = process_safety_attributes(response)

            log_success_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                response_text=response_text,
                id=prompt_dict.get("id", "NA"),
            )

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except IndexError as err:
            error_as_string = (
                f"Response is empty and blocked ({type(err).__name__} - {err})"
            )
            log_message = log_error_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
            )
            logging.info(
                f"Response is empty and blocked (i={index}, id={prompt_dict.get('id', 'NA')}) \nPrompt: {prompt[:50]}..."
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
            response_text = ""
            try:
                if len(response.candidates) == 0:
                    safety_attributes = BLOCKED_SAFETY_ATTRIBUTES
                else:
                    safety_attributes = process_safety_attributes(response)
            except:
                safety_attributes = BLOCKED_SAFETY_ATTRIBUTES

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict

        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file,
                    log_message=log_message,
                    log=True,
                )
            raise err

    async def _query_chat(self, prompt_dict: dict, index: int | str):
        """
        Async method for querying the model with a chat prompt
        (prompt_dict["prompt"] is a list of strings to sequentially send to the model),
        i.e. multi-turn chat with history.
        """
        prompt, model_name, client, generation_config, _ = (
            await self._obtain_model_inputs(
                prompt_dict=prompt_dict, system_instruction=None
            )
        )

        # chat = client.start_chat(history=[])
        chat = client.aio.chats.create(
            model=model_name,
            config=generation_config,
            history=[],
        )
        response_list = []
        safety_attributes_list = []
        try:
            for message_index, message in enumerate(prompt):
                # send the messages sequentially
                # run the predict method in a separate thread using run_in_executor
                response = await chat.send_message(
                    message=message,
                    config=generation_config,
                )
                response_text = process_response(response)
                safety_attributes = process_safety_attributes(response)

                response_list.append(response_text)
                safety_attributes_list.append(safety_attributes)

                log_success_response_chat(
                    index=index,
                    model=f"Gemini ({model_name})",
                    message_index=message_index,
                    n_messages=len(prompt),
                    message=message,
                    response_text=response_text,
                    id=prompt_dict.get("id", "NA"),
                )

            logging.info(
                f"Chat completed (i={index}, id={prompt_dict.get('id', 'NA')})"
            )

            prompt_dict["response"] = response_list
            prompt_dict["safety_attributes"] = safety_attributes_list
            return prompt_dict
        except IndexError as err:
            error_as_string = (
                f"Response is empty and blocked ({type(err).__name__} - {err})"
            )
            log_message = log_error_response_chat(
                index=index,
                model=f"Gemini ({model_name})",
                message_index=message_index,
                n_messages=len(prompt),
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
            )
            logging.info(
                f"Response is empty and blocked (i={index}, id={prompt_dict.get('id', 'NA')}) \nPrompt: {message[:50]}..."
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
            response_text = response_list + [""]
            try:
                if len(response.candidates) == 0:
                    safety_attributes = BLOCKED_SAFETY_ATTRIBUTES
                else:
                    safety_attributes = process_safety_attributes(response)
            except:
                safety_attributes = BLOCKED_SAFETY_ATTRIBUTES

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_chat(
                index=index,
                model=f"Gemini ({model_name})",
                message_index=message_index,
                n_messages=len(prompt),
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
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
        (prompt_dict["prompt"] is a list of dictionaries with keys "role" and "parts",
        where "role" is one of "user", "model" and "parts" is the message),
        i.e. multi-turn chat with history.
        """
        if prompt_dict["prompt"][0]["role"] == "system":
            prompt, model_name, client, generation_config, _ = (
                await self._obtain_model_inputs(
                    prompt_dict=prompt_dict,
                    system_instruction=prompt_dict["prompt"][0]["parts"],
                )
            )
            # Used to skip the system message in the prompt history
            first_user_idx = 1
        else:
            prompt, model_name, client, generation_config, _ = (
                await self._obtain_model_inputs(
                    prompt_dict=prompt_dict, system_instruction=None
                )
            )
            first_user_idx = 0

        chat = client.aio.chats.create(
            model=model_name,
            config=generation_config,
            history=[
                convert_history_dict_to_content(
                    content_dict=x,
                    media_folder=self.settings.media_folder,
                    client=client,
                )
                for x in prompt[first_user_idx:-1]
            ],
        )

        try:
            # No need to send the generation_config again, as it is no different
            # from the one used to create the chat
            last_msg = prompt[-1]
            print(f"whole prompt: {prompt}")
            print(f"last_msg: {last_msg}")
            # msg_to_send = convert_dict_to_input(
            #     content_dict=prompt[-1], media_folder=self.settings.media_folder
            # )

            msg_to_send = parse_parts(
                prompt[-1]["parts"],
                media_folder=self.settings.media_folder,
                client=client,
            )

            assert (
                len(msg_to_send) == 1
            ), "Only one message is allowed in the last message"
            msg_to_send = msg_to_send[0]

            print(f"msg_to_send: {msg_to_send}")

            response = await chat.send_message(
                # message=convert_dict_to_input(
                #     content_dict=prompt[-1], media_folder=self.settings.media_folder
                # ),
                message=msg_to_send
            )

            response_text = process_response(response)
            safety_attributes = process_safety_attributes(response)

            log_success_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                response_text=response_text,
                id=prompt_dict.get("id", "NA"),
            )

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except IndexError as err:
            error_as_string = (
                f"Response is empty and blocked ({type(err).__name__} - {err})"
            )
            log_message = log_error_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {prompt[:50]}..."
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
            response_text = ""
            try:
                if len(response.candidates) == 0:
                    safety_attributes = BLOCKED_SAFETY_ATTRIBUTES
                else:
                    safety_attributes = process_safety_attributes(response)
            except:
                safety_attributes = BLOCKED_SAFETY_ATTRIBUTES

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"Gemini ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
                id=prompt_dict.get("id", "NA"),
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
            elif (
                all(isinstance(message, dict) for message in prompt_dict["prompt"])
                and (
                    set(prompt_dict["prompt"][0].keys()) == {"role", "parts"}
                    and prompt_dict["prompt"][0]["role"]
                    in list(gemini_chat_roles) + ["system"]
                )
                and all(
                    [
                        set(d.keys()) == {"role", "parts"}
                        and d["role"] in gemini_chat_roles
                        for d in prompt_dict["prompt"][1:]
                    ]
                )
            ):
                return await self._query_history(
                    prompt_dict=prompt_dict,
                    index=index,
                )

        raise TYPE_ERROR
