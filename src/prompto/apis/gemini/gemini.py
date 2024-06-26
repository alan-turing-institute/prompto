import asyncio
import logging
import os
from typing import Any

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

from prompto.apis.base import AsyncBaseAPI
from prompto.apis.gemini.gemini_utils import (
    parse_multimedia,
    process_response,
    process_safety_attributes,
)
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    get_model_name_identifier,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)

PROJECT_VAR_NAME = "GEMINI_PROJECT_ID"
LOCATION_VAR_NAME = "GEMINI_LOCATION"
MODEL_NAME_VAR_NAME = "GEMINI_MODEL_NAME"


class AsyncGeminiAPI(AsyncBaseAPI):
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

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        """
        For Gemini, there are some optional variables:
        - GEMINI_PROJECT_ID
        - GEMINI_LOCATION
        - GEMINI_MODEL_NAME

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
        issues.extend(
            check_optional_env_variables_set(
                [PROJECT_VAR_NAME, LOCATION_VAR_NAME, MODEL_NAME_VAR_NAME]
            )
        )

        return issues

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        """
        For Gemini, we make the following model-specific checks:
        - "prompt" must be a string or a list of strings
        - if "model_name" is not in the prompt dictionary, then the default
          environment variables (GEMINI_PROJECT_ID, GEMINI_LOCATION, GEMINI_MODEL_NAME)
          must be set
        - if "model_name" is in the prompt dictionary, then the model-specific
          project and location environment variables (GEMINI_PROJECT_ID_{identifier},
          GEMINI_LOCATION_{identifier}) (where identifier is the model name with
          invalid characters replaced by underscores obtained using
          get_model_name_identifier function) can be optionally set.
          If neither the model-specific environment variable is set or the default
          environment variable is set, it is possible to run if you are signed in
          with gcloud CLI
        - if "safety_filter" is provided, check that it's one of the valid options
          ("none", "few", "some", "default", "most")
        - if "generation_config" is provided, check that it can create a valid
          vertexai.generative_models.GenerationConfig object

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

        # check prompt is of the right type (string or list of strings)
        match prompt_dict["prompt"]:
            case str(_):
                pass
            case [str(_)]:
                pass
            case _:
                issues.append(
                    TypeError(
                        f"if api == 'gemini', then prompt must be a string or a list, "
                        f"not {type(prompt_dict['prompt'])}"
                    )
                )

        if "model_name" not in prompt_dict:
            # use the default environment variables
            # check the required environment variables are set
            issues.extend(check_required_env_variables_set([MODEL_NAME_VAR_NAME]))

            # check the optional environment variables are set and warn if not
            issues.extend(
                check_optional_env_variables_set([PROJECT_VAR_NAME, LOCATION_VAR_NAME])
            )
        else:
            # use the model specific environment variables
            model_name = prompt_dict["model_name"]
            # replace any invalid characters in the model name
            identifier = get_model_name_identifier(model_name)

            # check the optional environment variables are set and warn if not
            issues.extend(
                check_optional_env_variables_set(
                    [
                        f"{PROJECT_VAR_NAME}_{identifier}",
                        PROJECT_VAR_NAME,
                        f"{LOCATION_VAR_NAME}_{identifier}",
                        LOCATION_VAR_NAME,
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
            except TypeError as err:
                issues.append(TypeError(f"Invalid generation_config parameter: {err}"))
            except Exception as err:
                issues.append(ValueError(f"Invalid generation_config parameter: {err}"))

        return issues

    async def _obtain_model_inputs(
        self, prompt_dict: dict
    ) -> tuple[str, str, dict, dict, list[Part] | None]:
        """
        Async method to obtain the model inputs from the prompt dictionary.

        Parameters
        ----------
        prompt_dict : dict
            The prompt dictionary to use for querying the model

        Returns
        -------
        tuple[str, str, dict, dict, list[Part] | None]
            A tuple containing the prompt, model name, safety settings,
            the generation config, and list of multimedia parts (if passed)
            to use for querying the model
        """
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
                async with FILE_WRITE_LOCK:
                    write_log_message(
                        log_file=self.log_file, log_message=log_message, log=True
                    )
                raise ValueError(log_message)

            project_id = PROJECT_VAR_NAME
            location_id = LOCATION_VAR_NAME
        else:
            # use the model specific environment variables if they exist
            # replace any invalid characters in the model name
            identifier = get_model_name_identifier(model_name)

            project_id = f"{PROJECT_VAR_NAME}_{identifier}"
            if project_id not in os.environ:
                project_id = PROJECT_VAR_NAME

            location_id = f"{LOCATION_VAR_NAME}_{identifier}"
            if location_id not in os.environ:
                location_id = LOCATION_VAR_NAME

        PROJECT_ID = os.environ.get(PROJECT_VAR_NAME, None)
        LOCATION_ID = os.environ.get(LOCATION_VAR_NAME, None)

        # initialise the vertexai project
        vertexai.init(project=PROJECT_ID, location=LOCATION_ID)

        # define safety settings
        safety_filter = prompt_dict.get("safety_filter", None)
        if safety_filter is None:
            safety_filter = "default"

        # explicitly set the safety settings
        if safety_filter == "none":
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        elif safety_filter == "few":
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        elif safety_filter in ["default", "some"]:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        elif safety_filter == "most":
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        else:
            raise ValueError(
                f"safety_filter '{safety_filter}' not recognised. Must be one of: "
                f"none', 'few', 'default'/'some', 'most'"
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
            "max_output_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 24,
        }
        for key, value in default_generation_config.items():
            if key not in generation_config:
                generation_config[key] = value

        # parse multimedia data (if any)
        multimedia_dict = prompt_dict.get("multimedia", None)
        if multimedia_dict is not None:
            multimedia = parse_multimedia(
                multimedia_dict, media_folder=self.settings.media_folder
            )
        else:
            multimedia = None

        return prompt, model_name, safety_settings, generation_config, multimedia

    async def _async_query_string(self, prompt_dict: dict, index: int | str):
        """
        Async method for querying the model with a string prompt
        (prompt_dict["prompt"] is a string),
        i.e. single-turn completion or chat.
        """
        prompt, model_name, safety_settings, generation_config, multimedia = (
            await self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        # prepare the contents to send to the model
        if multimedia is not None:
            # prepend the multimedia to the prompt
            contents = multimedia + [Part.from_text(prompt)]
        else:
            contents = [Part.from_text(prompt)]

        try:

            def run_predict():
                return GenerativeModel(model_name).generate_content(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )

            # run the predict method in a separate thread using run_in_executor
            response = await asyncio.get_event_loop().run_in_executor(None, run_predict)
            response_text = process_response(response)
            safety_attributes = process_safety_attributes(response)

            log_success_response_query(
                index=index,
                model=f"gemini ({model_name})",
                prompt=prompt,
                response_text=response_text,
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
                model=f"gemini ({model_name})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {prompt[:50]}..."
            )
            if isinstance(err, IndexError):
                async with FILE_WRITE_LOCK:
                    write_log_message(
                        log_file=self.log_file, log_message=log_message, log=True
                    )
                response_text = ""
                if len(response.candidates) == 0:
                    safety_attributes = {
                        "blocked": "True",
                        "finish_reason": "block_reason: OTHER",
                    }
                else:
                    safety_attributes = process_safety_attributes(response)

                prompt_dict["response"] = response_text
                prompt_dict["safety_attributes"] = safety_attributes
                return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"gemini ({model_name})",
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

    async def _async_query_chat(self, prompt_dict: dict, index: int | str):
        """
        Async method for querying the model with a chat prompt
        (prompt_dict["prompt"] is a list of strings to sequentially send to the model),
        i.e. multi-turn chat with history.
        """
        prompt, model_name, safety_settings, generation_config, _ = (
            await self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        model = GenerativeModel(model_name)
        chat = model.start_chat(history=[])

        def run_predict(message):
            return chat.send_message(
                content=message,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False,
            )

        async def send_message(message):
            return await asyncio.get_event_loop().run_in_executor(
                None,
                run_predict,
                message,
            )

        response_list = []
        safety_attributes_list = []
        try:
            for message_index, message in enumerate(prompt):
                # send the messages sequentially
                # run the predict method in a separate thread using run_in_executor
                response = await send_message(message=message)
                response_text = process_response(response)
                safety_attributes = process_safety_attributes(response)

                response_list.append(response_text)
                safety_attributes_list.append(safety_attributes)

                log_success_response_chat(
                    index=index,
                    model=f"gemini ({model_name})",
                    message_index=message_index,
                    n_messages=len(prompt),
                    message=message,
                    response_text=response_text,
                )

            logging.info(f"Chat completed (i={index})")

            prompt_dict["response"] = response_list
            prompt_dict["safety_attributes"] = safety_attributes_list
            return prompt_dict
        except IndexError as err:
            error_as_string = (
                f"Response is empty and blocked ({type(err).__name__} - {err})"
            )
            log_message = log_error_response_chat(
                index=index,
                model=f"gemini ({model_name})",
                message_index=message_index,
                n_messages=len(prompt),
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {message[:50]}..."
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
            response_text = ""
            if len(response.candidates) == 0:
                safety_attributes = {
                    "blocked": "True",
                    "finish_reason": "block_reason: OTHER",
                }
            else:
                safety_attributes = process_safety_attributes(response)

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_chat(
                index=index,
                model=f"gemini ({model_name})",
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
            case _:
                pass

        raise TypeError(
            f"if api == 'gemini', then prompt must be a string or a list, "
            f"not {type(prompt_dict['prompt'])}"
        )
