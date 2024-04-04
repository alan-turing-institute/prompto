import vertexai
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
from vertexai.preview.generative_models import GenerativeModel, Image, Part

HarmCategory = gapic_content_types.HarmCategory
HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold

import asyncio
import logging
import os
from typing import Any

from batch_llm import Experiment, Settings
from batch_llm.base import BaseModel
from batch_llm.models.gemini.gemini_utils import parse_multimedia
from src.batch_llm.utils import (
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    write_log_message,
)


class Gemini(BaseModel):
    def __init__(
        self, settings: Settings, experiment: Experiment, *args: Any, **kwargs: Any
    ):
        self.settings: Settings = settings
        self.experiment: Experiment = experiment

    def _obtain_model_inputs(self, prompt_dict: dict) -> tuple:
        prompt = prompt_dict["prompt"]

        model_id = prompt_dict.get("model_name", None) or os.environ.get("MODEL_ID")
        if model_id is None:
            log_message = "MODEL_ID environment variable not set"
            write_log_message(
                log_file=self.experiment.log_file, log_message=log_message, log=True
            )

        # define safety settings
        safety_filter = prompt_dict.get("safety_filter", None)
        if safety_filter is None:
            raise ValueError("safety_filter must be set")

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
            raise ValueError(f"safety_filter '{safety_filter}' not recognised")

        # get parameters dict (if any)
        generation_config = prompt_dict.get("parameters", None)
        if generation_config is None:
            generation_config = {}
        if type(generation_config) is not dict:
            raise TypeError(
                f"parameters must be a dictionary, not {type(generation_config)}"
            )

        # add in default parameters
        # (default for model-b but capped at 1024 tokens so works for model-a)
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

        return prompt, model_id, safety_settings, generation_config, multimedia

    @classmethod
    def _process_response(response):
        response_text = response.candidates[0].content.parts[0].text
        return response_text

    @classmethod
    def _process_safety_attributes(response):
        safety_attributes = {
            x.category.name: str(x.probability)
            for x in response.candidates[0].safety_ratings
        }
        # list of booleans indicating whether each category is blocked
        safety_attributes["blocked"] = str(
            [x.blocked for x in response.candidates[0].safety_ratings]
        )
        safety_attributes["finish_reason"] = str(
            response.candidates[0].finish_reason.name
        )

        return safety_attributes

    def _query_string(self, prompt_dict: dict, index: int | str):
        prompt, model_id, safety_settings, generation_config, multimedia = (
            self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        # prepare the contents to send to the model
        if multimedia is not None:
            # prepend the multimedia to the prompt
            contents = multimedia + [Part.from_text(prompt)]
        else:
            contents = [Part.from_text(prompt)]

        try:
            response = GenerativeModel(model_id).generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False,
            )
            response_text = self.process_response(response)
            safety_attributes = self.process_safety_attributes(response)

            log_success_response_query(
                index=index,
                model=f"gemini ({model_id})",
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
                model=f"gemini ({model_id})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {prompt[:50]}..."
            )
            if isinstance(err, IndexError):
                write_log_message(
                    log_file=self.experiment.log_file, log_message=log_message, log=True
                )
                response_text = ""
                if len(response.candidates) == 0:
                    safety_attributes = {
                        "blocked": "True",
                        "finish_reason": "block_reason: OTHER",
                    }
                else:
                    safety_attributes = self.process_safety_attributes(response)

                prompt_dict["response"] = response_text
                prompt_dict["safety_attributes"] = safety_attributes
                return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"gemini ({model_id})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.experiment.log_file,
                log_message=log_message,
                log=True,
            )
            raise err

    def _query_chat(self, prompt_dict: dict, index: int | str):
        prompt, model_id, safety_settings, generation_config, multimedia = (
            self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        model = GenerativeModel(model_id)
        chat = model.start_chat(history=[])

        response_list = []
        safety_attributes_list = []
        try:
            for message_index, message in enumerate(prompt):
                # send the messages sequentially
                # run the predict method in a separate thread using run_in_executor
                response = chat.send_message(
                    content=message,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                response_text = self.process_response(response)
                safety_attributes = self.process_safety_attributes(response)

                response_list.append(response_text)
                safety_attributes_list.append(safety_attributes)

                log_success_response_chat(
                    index=index,
                    model=f"gemini ({model_id})",
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
                model=f"gemini ({model_id})",
                message_index=message_index,
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {message[:50]}..."
            )

            write_log_message(
                log_file=self.experiment.log_file, log_message=log_message, log=True
            )
            response_text = ""
            if len(response.candidates) == 0:
                safety_attributes = {
                    "blocked": "True",
                    "finish_reason": "block_reason: OTHER",
                }
            else:
                safety_attributes = self.process_safety_attributes(response)

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_chat(
                index=index,
                model=f"gemini ({model_id})",
                message_index=message_index,
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.experiment.log_file,
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
                f"If model == 'gemini', then prompt must be a string or a list, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict

    async def _async_query_string(self, prompt_dict: dict, index: int | str):
        prompt, model_id, safety_settings, generation_config, multimedia = (
            self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        # prepare the contents to send to the model
        if multimedia is not None:
            # prepend the multimedia to the prompt
            contents = multimedia + [Part.from_text(prompt)]
        else:
            contents = [Part.from_text(prompt)]

        try:

            def run_predict():
                return GenerativeModel(model_id).generate_content(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )

            # run the predict method in a separate thread using run_in_executor
            response = await asyncio.get_event_loop().run_in_executor(None, run_predict)
            response_text = self.process_response(response)
            safety_attributes = self.process_safety_attributes(response)

            log_success_response_query(
                index=index,
                model=f"gemini ({model_id})",
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
                model=f"gemini ({model_id})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {prompt[:50]}..."
            )
            if isinstance(err, IndexError):
                write_log_message(
                    log_file=self.experiment.log_file, log_message=log_message, log=True
                )
                response_text = ""
                if len(response.candidates) == 0:
                    safety_attributes = {
                        "blocked": "True",
                        "finish_reason": "block_reason: OTHER",
                    }
                else:
                    safety_attributes = self.process_safety_attributes(response)

                prompt_dict["response"] = response_text
                prompt_dict["safety_attributes"] = safety_attributes
                return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_query(
                index=index,
                model=f"gemini ({model_id})",
                prompt=prompt,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.experiment.log_file,
                log_message=log_message,
                log=True,
            )
            raise err

    async def _async_query_chat(self, prompt_dict: dict, index: int | str):
        prompt, model_id, safety_settings, generation_config, multimedia = (
            self._obtain_model_inputs(prompt_dict=prompt_dict)
        )

        model = GenerativeModel(model_id)
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
                response_text = self.process_response(response)
                safety_attributes = self.process_safety_attributes(response)

                response_list.append(response_text)
                safety_attributes_list.append(safety_attributes)

                log_success_response_chat(
                    index=index,
                    model=f"gemini ({model_id})",
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
                model=f"gemini ({model_id})",
                message_index=message_index,
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            logging.info(
                f"Response is empty and blocked (i={index}) \nPrompt: {message[:50]}..."
            )

            write_log_message(
                log_file=self.experiment.log_file, log_message=log_message, log=True
            )
            response_text = ""
            if len(response.candidates) == 0:
                safety_attributes = {
                    "blocked": "True",
                    "finish_reason": "block_reason: OTHER",
                }
            else:
                safety_attributes = self.process_safety_attributes(response)

            prompt_dict["response"] = response_text
            prompt_dict["safety_attributes"] = safety_attributes
            return prompt_dict
        except Exception as err:
            error_as_string = f"{type(err).__name__} - {err}"
            log_message = log_error_response_chat(
                index=index,
                model=f"gemini ({model_id})",
                message_index=message_index,
                message=message,
                responses_so_far=response_list,
                error_as_string=error_as_string,
            )
            write_log_message(
                log_file=self.experiment.log_file,
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
                f"If model == 'gemini', then prompt must be a string or a list, "
                f"not {type(prompt_dict['prompt'])}"
            )

        return response_dict
